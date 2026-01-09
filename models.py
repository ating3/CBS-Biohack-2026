import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertModel, BertConfig
from preprocess import *

class DNATokenizer():
    pass

class GenomicTransformerDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, bio_features=None):
        super().__init__()
        self.sequences = list(sequences.iloc[:, 0])
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.tokenized = [tokenizer(seq) for seq in self.sequences]
        self.bio_features = bio_features
        self.has_bio_features = bio_features is not None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        encoding = self.tokenized[index]
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        
        item = {
            'input_ids': input_ids,
            'labels': self.labels[index]
        }
        
        if self.has_bio_features:
            item['bio_features'] = self.bio_features[index]
        
        return item

class GenomicCNNDataset(Dataset):
    def __init__(self, sequences, labels, bio_features=None):
        super().__init__()
        self.sequences = list(sequences.iloc[:, 0])
        self.labels = labels.values
        self.bio_features = bio_features
        self.has_bio_features = bio_features is not None

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        item = {'input_ids': torch.tensor(one_hot_dna(self.sequences[index]), dtype=torch.float32), 'labels': self.labels[index]}
        if self.has_bio_features:
            item['bio_features'] = self.bio_features[index]
        return item

class CNNTransformer(nn.Module):
    def __init__(self, num_outputs, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # ============ CNN Feature Extraction ============
        # Multi-scale convolutional layers to detect motifs of different lengths
        self.conv1_short = nn.Conv1d(4, 64, kernel_size=3, padding=1)   # 3bp motifs
        self.conv1_med = nn.Conv1d(4, 64, kernel_size=7, padding=3)     # 7bp motifs
        self.conv1_long = nn.Conv1d(4, 64, kernel_size=15, padding=7)   # 15bp motifs
        
        self.bn1 = nn.BatchNorm1d(192)  # 64*3 = 192
        
        # Deeper CNN layers
        self.conv2 = nn.Conv1d(192, 256, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, d_model, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(d_model)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(0.2)
        
        # ============ Transformer Encoder ============        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Important: [batch, seq, features]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ============ Classification Head ============
        # CLS token approach (like BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Alternative: use both CLS token and global pooling
        self.fc1 = nn.Linear(d_model * 2, 512)  # *2 for CLS + pooling
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_outputs)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # ============ CNN Feature Extraction ============
        # Multi-scale convolution
        x_short = F.relu(self.conv1_short(x))
        x_med = F.relu(self.conv1_med(x))
        x_long = F.relu(self.conv1_long(x))
        
        x = torch.cat([x_short, x_med, x_long], dim=1)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        # x shape: [batch, d_model, seq_len]
        
        # ============ Prepare for Transformer ============
        # Transpose to [batch, seq_len, d_model] for transformer
        x = x.transpose(1, 2)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]
        
        # ============ Transformer Encoding ============
        x = self.transformer_encoder(x)
        # x shape: [batch, seq_len+1, d_model]
        
        # ============ Classification ============
        # Extract CLS token
        cls_output = x[:, 0, :]  # [batch, d_model]
        
        # Global average pooling over sequence (excluding CLS token)
        sequence_output = x[:, 1:, :].mean(dim=1)  # [batch, d_model]
        
        # Concatenate both representations
        x = torch.cat([cls_output, sequence_output], dim=1)  # [batch, d_model*2]
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class GenomicAttentionCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        
        # Multi-scale convolutional layers
        self.conv1_short = nn.Conv1d(4, 32, kernel_size=4, padding=2)
        self.conv1_med = nn.Conv1d(4, 32, kernel_size=8, padding=4)
        self.conv1_long = nn.Conv1d(4, 32, kernel_size=16, padding=8)
        
        self.bn1 = nn.BatchNorm1d(96)
        
        self.conv2 = nn.Conv1d(96, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=8, padding=4)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.dropout_heavy = nn.Dropout(0.5)
        
        # Attention mechanism - use adaptive pooling to ensure consistent output
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_outputs)
        
    def forward(self, x):
        # Multi-scale feature extraction
        x_short = F.relu(self.conv1_short(x))
        x_med = F.relu(self.conv1_med(x))
        x_long = F.relu(self.conv1_long(x))
        
        x = torch.cat([x_short, x_med, x_long], dim=1)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Channel attention (Squeeze-and-Excitation style)
        attention_weights = self.attention(x)  # Shape: [batch, 256, 1]
        x = x * attention_weights  # Broadcast attention across spatial dimension
        
        # Global average pooling
        x = self.global_pool(x)  # Shape: [batch, 256, 1]
        x = x.squeeze(-1)  # Shape: [batch, 256]
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_heavy(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
class GenomicAttentionCNN2(nn.Module):
    def __init__(self, num_outputs, bio_feature_dim=5):
        super().__init__()
        
        # Multi-scale convolutional layers
        self.conv1_short = nn.Conv1d(4, 32, kernel_size=4, padding=2)
        self.conv1_med = nn.Conv1d(4, 32, kernel_size=8, padding=4)
        self.conv1_long = nn.Conv1d(4, 32, kernel_size=16, padding=8)
        
        self.bn1 = nn.BatchNorm1d(96)
        
        self.conv2 = nn.Conv1d(96, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=8, padding=4)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.dropout_heavy = nn.Dropout(0.5)
        
        # Attention mechanism - use adaptive pooling to ensure consistent output
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
            nn.Conv1d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Bio feature processing pathway
        self.bio_fc = nn.Sequential(
            nn.Linear(bio_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Classifier with combined features
        combined_dim = 256 + 128  # CNN features + bio features
        self.fc1 = nn.Linear(combined_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_outputs)
        
    def forward(self, x, bio_features):
        # Multi-scale feature extraction
        x_short = F.relu(self.conv1_short(x))
        x_med = F.relu(self.conv1_med(x))
        x_long = F.relu(self.conv1_long(x))
        
        x = torch.cat([x_short, x_med, x_long], dim=1)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Channel attention (Squeeze-and-Excitation style)
        attention_weights = self.attention(x)  # Shape: [batch, 256, 1]
        x = x * attention_weights  # Broadcast attention across spatial dimension
        
        # Global average pooling
        x = self.global_pool(x)  # Shape: [batch, 256, 1]
        x = x.squeeze(-1)  # Shape: [batch, 256]
        
        # Process bio features
        bio_output = self.bio_fc(bio_features)  # Shape: [batch, 128]
        
        # Combine CNN features with bio features
        combined = torch.cat([x, bio_output], dim=1)  # Shape: [batch, 384]
        
        # Final classification layers
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout_heavy(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
class GenomicCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=8, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)  # Global average pooling

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ChromatinBERT(nn.Module):
    def __init__(self, vocab_size, num_outputs=18, bio_features_dim=5):
        super(ChromatinBERT, self).__init__()

        # Configure BERT for DNA
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=6,  # Lighter than full BERT-base (12 layers)
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        # Initialize BERT from scratch (no pre-training, trained on your data)
        self.bert = BertModel(config)

        # Biological features branch
        self.bio_fc = nn.Sequential(
            nn.Linear(bio_features_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 512),  # BERT output + bio features
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_outputs)
        )

    def forward(self, input_ids, attention_mask, bio_features):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        bert_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        # Process biological features
        bio_output = self.bio_fc(bio_features)

        # Concatenate and classify
        combined = torch.cat([bert_output, bio_output], dim=1)
        logits = self.classifier(combined)

        return logits

class GenomicTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            num_outputs=18,
            max_len=200,
            dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim, #try 2 or 3
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Embedding(max_len + 1, embed_dim)

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(embed_dim//2, num_outputs)
        )
        
        #nn.Linear(embed_dim, num_outputs)

    def forward(self,x):
        B,L = x.shape
        x = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        positions = torch.arange(0, L+1, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        x = self.transformer(x)
    
        x = x[:, 0]
        #x = x.mean(dim=1)
        #no longer using mean pooling

        logits = self.classifier(x)
        return logits

class GenomicTransformer2Advanced(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            num_outputs=18,
            max_len=200,
            dropout=0.3,
            bio_feature_dim=5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bio features as learnable tokens
        self.bio_projection = nn.Linear(bio_feature_dim, embed_dim)
        self.bio_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-attention to integrate bio features with sequence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_outputs)
        )

    def forward(self, x, bio_features):
        B, L = x.shape
        
        # Sequence processing
        x = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoding[:, :L+1, :]
        x = self.transformer(x)
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # Bio feature processing
        bio_embed = self.bio_projection(bio_features).unsqueeze(1)  # [B, 1, embed_dim]
        bio_token = self.bio_token.expand(B, -1, -1)
        bio_repr = bio_token + bio_embed
        
        # Cross-attention: bio features attend to sequence
        bio_attended, _ = self.cross_attention(
            bio_repr, x, x
        )  # [B, 1, embed_dim]
        bio_attended = bio_attended.squeeze(1)  # [B, embed_dim]
        
        # Combine both representations
        combined = torch.cat([cls_output, bio_attended], dim=1)
        logits = self.classifier(combined)
        return logits

class GenomicTransformer2(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            num_outputs=18,
            max_len=200,
            dropout=0.3,
            bio_feature_dim=5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim, #try 2 or 3
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.bio_fc = nn.Sequential(
                nn.Linear(bio_feature_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(nn.Linear(embed_dim + 128, embed_dim + 128 // 2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(embed_dim + 128//2, num_outputs)
        )
        
        #nn.Linear(embed_dim, num_outputs)

    def forward(self, x, bio_features):
        B,L = x.shape
        x = self.embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        x = x[:, 0]

        bio_output = self.bio_fc(bio_features)
        combined = torch.cat([bio_output, x], dim=1)    
        
        logits = self.classifier(combined)
        return logits