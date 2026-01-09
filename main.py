import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from preprocess import *
from models import *

#python main.py -t GenomicAttentionCNN2 -e 50
#python main.py -t GenomicTransformer2Advanced -e 50

MODEL_REGISTRY = {
    "GenomicTransformer": GenomicTransformer, #standard transformer implementation

    "GenomicTransformer2": GenomicTransformer2, #includes additional FCNN to learn from bio_features as well

    "CNNTransformer": CNNTransformer, #CNN -> transformer architecture
    "GenomicCNN" : GenomicCNN, #standard CNN implementation
    "GenomicAttentionCNN": GenomicAttentionCNN, #CNN with convolutional layers of varying lengths to learn variable length dependencies
    "GenomicAttentionCNN2" : GenomicAttentionCNN2, #improved with incorporation of bio_features
    "ChromatinBERT": ChromatinBERT,
    "GenomicTransformer2Advanced" : GenomicTransformer2Advanced #incorporates cross attention to better integrate bio_features
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', help="model architecture to train on", required=True)
    parser.add_argument('-e', '--epochs', help="number of epochs to train on", required=True)
    args = parser.parse_args()
    #Load Data
    train_labels = pd.read_csv("trainlabels.csv", header=None)
    train_seq = pd.read_csv("trainsequences.csv", header=None)

    #augment dataset with reverse complement
    aug_train_seq, aug_train_labels = augment_sequences(train_seq[0].values, train_labels.values)
    #tokenizers
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    vocab_size = tokenizer.vocab_size

    model_type = args.model_type


    if model_type in ["GenomicCNN", "GenomicAttentionCNN", "CNNTransformer", "GenomicAttentionCNN2"]: #CNN and CNNTransformer
        model = MODEL_REGISTRY[args.model_type](num_outputs=18)

        if model_type == "GenomicAttentionCNN2":
            bio_features = [extract_sequence_features(row['sequence']) for _, row in aug_train_seq.iterrows()]
            dataset = GenomicCNNDataset(aug_train_seq, aug_train_labels, bio_features)
        else:
            dataset = GenomicCNNDataset(aug_train_seq, aug_train_labels)

    else: #Transformers with and without biofeatures
        model = MODEL_REGISTRY[args.model_type](num_outputs=18, vocab_size = vocab_size)

        if model_type in ["GenomicTransformer2", "GenomicTransformer2Advanced"]:
            bio_features = [extract_sequence_features(row['sequence']) for _, row in aug_train_seq.iterrows()]
            dataset = GenomicTransformerDataset(aug_train_seq, aug_train_labels, tokenizer, bio_features)
        else:
            dataset = GenomicTransformerDataset(aug_train_seq, aug_train_labels, tokenizer)        

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        correct = 0
        total = 0

        running_correct = 0.
        running_total = 0.


        for i, data in enumerate(train_dataloader):
            inputs = data['input_ids']
            labels = data['labels']
            inputs, labels = inputs.to(device), (labels.squeeze()-1).long().to(device)
            #squeeze() to remove excess dimension for label (batchsize, 1) -> (batchsize,)

            if model_type in ["GenomicTransformer2", "GenomicTransformer2Advanced", "GenomicAttentionCNN2"]:
                bio_features = data['bio_features']
                bio_features =  bio_features.to(device)
                outputs = model(inputs, bio_features)
            else:
                outputs = model(inputs)

            optimizer.zero_grad()

            loss = loss_fn(outputs,labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()


            #per 1000 batch level logging
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            #epoch level accuracy logging
            running_correct += (pred == labels).sum().item()
            running_total += labels.size(0)

            if i % 1000 == 999:
                last_loss = running_loss / 1000
                running_acc = correct / total
                print('  batch {} loss: {} accuracy: {}'.format(i + 1, last_loss, running_acc))
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', running_acc, tb_x)
                running_loss = 0.
                correct = 0
                total = 0

        epoch_acc = running_correct / running_total
        return last_loss, epoch_acc

    def train_loop(epochs=20):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/genomic_transformer2_advanced{}'.format(timestamp))
        epoch_number = 0


        best_vloss = 1_000_000.

        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)

            avg_loss, avg_acc = train_one_epoch(epoch_number, writer)


            running_vloss = 0.0
            vcorrect = 0
            vtotal = 0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            with torch.no_grad():
                for i, vdata in enumerate(val_dataloader):
                    vinputs = vdata['input_ids']
                    vlabels = vdata['labels']                
                    vinputs, vlabels,  = vinputs.to(device), (vlabels.squeeze()-1).long().to(device)

                    if model_type in ["GenomicTransformer2", "GenomicTransformer2Advanced", "GenomicAttentionCNN2"]:
                        vbio_features = vdata['bio_features']
                        vbio_features = vbio_features.to(device)
                        voutputs = model(vinputs,vbio_features)
                    else:
                        voutputs = model(vinputs)
                    
                    pred = voutputs.argmax(dim=1)
                    vcorrect += (pred==vlabels).sum().item()
                    vtotal += vlabels.size(0)

                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            avg_vacc = vcorrect/vtotal

            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('ACC train {} valid {}'.format(avg_acc, avg_vacc))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            writer.add_scalars('Training vs. Validation Accuracy',
                        {'Training': avg_acc, 'Validation': avg_vacc},
                        epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1

    train_loop(int(args.epochs))
