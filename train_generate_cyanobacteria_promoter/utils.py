import numpy as np
import os
import pandas as pd
from collections import Counter

# Function to get the frequency of each k-mer
def get_kmer_frequencies(kmers):
    kmer_counts = Counter(kmers)
    total_kmers = sum(kmer_counts.values())
    # Normalize the counts by the total number of kmers to get frequency
    kmer_freq = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freq

# Function to calculate k-mers of a sequence
def calculate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence)-k+1)]

def calculate_overall_kmer_correlation(dataset1, dataset2, k, flag=False):
    # Step 1: Generate a list of all kmers for each dataset
    kmers_dataset1 = [calculate_kmers(seq, k) for seq in dataset1]
    kmers_dataset2 = [calculate_kmers(seq, k) for seq in dataset2]
    
    # Step 2: Flatten the list of kmer lists to get a single list of kmers for each dataset
    flat_kmers_dataset1 = [kmer for sublist in kmers_dataset1 for kmer in sublist]
    flat_kmers_dataset2 = [kmer for sublist in kmers_dataset2 for kmer in sublist]
    
    # Step 3: Get the frequency of each kmer in each dataset
    freq_dataset1 = get_kmer_frequencies(flat_kmers_dataset1)
    freq_dataset2 = get_kmer_frequencies(flat_kmers_dataset2)
    
    # Step 4: Create pandas series for each frequency distribution, ensuring a common index
    s1 = pd.Series(freq_dataset1).fillna(0)
    s2 = pd.Series(freq_dataset2).fillna(0)
    common_index = s1.index.union(s2.index)
    s1 = s1.reindex(common_index, fill_value=0)
    s2 = s2.reindex(common_index, fill_value=0)
    
    # Step 5: Compute the Pearson correlation for the kmers
    correlation = s1.corr(s2)
    
    if flag :
        return s1,s2,correlation
    
    else:
        return correlation
    

def make_fasta_file(sequences,path): 


    if not os.path.exists(path):

        with open(path, 'w') as f:
            f.write("This is a new file.")

        print(f"文件 {path} 不存在，已创建新文件。")
    else:
        print(f"文件 {path} 已存在。")

    with open(path, 'w') as file:
            for i, seq in enumerate(sequences, start=1):
                seq = seq.upper()
                file.write(f'>Sequence_{i}\n')  # 写入序列标识符
                file.write(f'{seq}\n')  # 写入序列

    print(f"File {path} created and sequences written successfully.")


def decode_one_hot(one_hot_array):
        
        base_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        decoded_sequence = []
        
        for row in range(one_hot_array.shape[1]):

            max_index = np.argmax(one_hot_array[:,row])
            base = base_mapping[max_index]
            decoded_sequence.append(base)

        return ''.join(decoded_sequence)

def one_hot_encoding(sequence):
    
    bases = ['A', 'C', 'G', 'T']
    base_dict = dict(zip(bases, range(4)))
    
    sequence = sequence.upper() #全部转化为大写
    
    length = len(sequence)
    encoded_sequence = np.zeros((4, length), dtype=int)

    for i, base in enumerate(sequence):
        if base in base_dict:
            encoded_sequence[base_dict[base], i] = 1

    return encoded_sequence

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))