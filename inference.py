import argparse
import pandas as pd
from models import *
from preprocess import extract_sequence_features
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from main import MODEL_REGISTRY
#12/31 accuracy 16%: python ./inference.py -i /home/ating3/dev/CBS_Biohack_2025/testsequences.csv -o inferencing -m model_20251231_225037_4 -t GenomicTransformer2
#1/1 accuracy 17.5%: python ./inference.py -i /home/ating3/dev/CBS_Biohack_2025/testsequences.csv -o inferencing -m model_20260102_001026_6 -t GenomicCNN
#1/2 accuracy 19.078%: python ./inference.py -i /home/ating3/dev/CBS_Biohack_2025/testsequences.csv -o inferencing -m model_20260102_122257_47 -t GenomicAttentionCNN
#1/2 accuracy 18.9%: python ./inference.py -i /home/ating3/dev/CBS_Biohack_2025/testsequences.csv -o inferencing -m model_20260102_141647_12 -t CNNTransformer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to csv file for inferencing", required=True)
    parser.add_argument('-o', '--output', help="output directory", required=True)
    parser.add_argument('-m', '--model_weights', help="path to model weights", required=True)
    parser.add_argument('-t', '--model_type', help="model architecture to inference on", required=True)

    device = "cpu"
    #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    test_seq = pd.read_csv(args.input)
    test_seq.columns = ["sequence"]
    weights_path = args.model_weights

    def collate_fn(seqs):
        return torch.stack([
            torch.tensor(one_hot_dna(s), dtype=torch.float32)
            for s in seqs
        ])

    #tokenizer
    model_type = args.model_type
    if model_type in ["GenomicCNN", "GenomicAttentionCNN", "CNNTransformer"]:
        #CNN implementation
        model = MODEL_REGISTRY[args.model_type](num_outputs=18)

        dataloader = DataLoader(
            test_seq["sequence"].tolist(),
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )

    else: 
        #transformer implementations
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        vocab_size = tokenizer.vocab_size
        model = MODEL_REGISTRY[args.model_type](num_classes=18, vocab_size=vocab_size)
        if model_type == "GenomicTransformer2":
            bio_features = torch.tensor([extract_sequence_features(row['sequence']) for _, row in test_seq.iterrows()], dtype=torch.float32).to(device)
        
        input_ids = torch.stack(
            test_seq["sequence"].apply(lambda x: tokenizer(x, return_tensors="pt")["input_ids"].squeeze(0)).tolist()
        ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            if args.model_type == "GenomicTransformer2":
                outputs = model(input_ids, bio_features)
            else:
                outputs = model(batch)
                
            preds = torch.argmax(outputs, dim=1).cpu()
            predictions.append(preds)

    predictions = torch.cat(predictions).numpy() + 1
    output = pd.DataFrame(predictions.astype(int))
    output.to_csv(f'{args.output}/{args.model_type}_prediction.csv', index=False)

