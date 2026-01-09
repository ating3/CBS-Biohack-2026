import numpy as np
import pandas as pd

#Preprocessing
def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([complement.get(base, base) for base in reversed(seq)])

def augment_sequences(sequences, labels):
    """double training set by giving reverse complement of DNA as well"""
    aug_seqs = []
    aug_labels = []
    for seq, label in zip(sequences, labels):
        aug_seqs.append(seq)
        aug_labels.append(label)
        aug_seqs.append(reverse_complement(seq))
        aug_labels.append(label)
    return pd.DataFrame(aug_seqs, columns=["sequence"]), pd.DataFrame(np.array(aug_labels),  columns=["labels"])

def one_hot_dna(seq):
    nt_to_idx = {'A':0, 'T':1, 'G':2, 'C':3}
    onehot = np.zeros([4, len(seq)], dtype=np.float32)
    for i,c in enumerate(seq):
        onehot[nt_to_idx[c], i] = 1.0
    return onehot

def extract_sequence_features(sequence):
    """Extract bio features"""
    features = []

    # GC content
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    features.append(gc_content)

    # CpG islands
    cpg_count = sum(1 for i in range(len(sequence)-1) if sequence[i:i+2] == 'CG')
    cpg_density = cpg_count / (len(sequence) - 1) if len(sequence) > 1 else 0
    features.append(cpg_density)

    # Repeat density
    repeat_bases = 0
    for k in [2, 3, 4]:
        for i in range(len(sequence) - 2*k):
            kmer = sequence[i:i+k]
            if i+2*k <= len(sequence) and sequence[i+k:i+2*k] == kmer:
                repeat_bases += k
    repeat_density = repeat_bases / len(sequence) if len(sequence) > 0 else 0
    features.append(repeat_density)

    # Homopolymer runs
    max_homopolymer = 0
    if sequence:
        current_base = sequence[0]
        current_run = 1
        for base in sequence[1:]:
            if base == current_base:
                current_run += 1
                max_homopolymer = max(max_homopolymer, current_run)
            else:
                current_base = base
                current_run = 1
    features.append(max_homopolymer / len(sequence) if len(sequence) > 0 else 0)

    # TATA box
    tata_variants = ['TATAAA', 'TATAA', 'TATA']
    has_tata = any(motif in sequence for motif in tata_variants)
    features.append(float(has_tata))

    return np.array(features, dtype=np.float32)