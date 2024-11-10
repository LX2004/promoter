import scipy as sp
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import csv
import os

def write_good_record(dict1,dict2,file_path):

    merged_dict = {**dict1, **dict2}

    if not os.path.isfile(file_path):
        with open(file_path, 'w') as file:
            file.write(f"{merged_dict}\n")
    else:
        with open(file_path, 'a') as file:
            file.write(f"{merged_dict}\n")

def one_hot(sequence):
    
    bases = ['A','T','G','C']
    one_hot_encoded = np.zeros((len(sequence), len(bases)))

    for i, base in enumerate(sequence):
        one_hot_encoded[i, bases.index(base)] = 1
        
    return one_hot_encoded


def loss_pierxun(output,target):

    target_mean = torch.mean(target)
    outpu_mean = torch.mean(output)

    target_var = torch.std(target)
    output_var = torch.std(output)

    p = torch.mean( (output - outpu_mean) * (target - target_mean) )

    if output_var == 0:
        
        p /= ((output_var + 1e-7) * target_var)
        return p

    p /= (output_var * target_var)

    # print('皮尔逊相关系数：', p)

    return p

def text_build_vocab():
    
    dic = [a for a in 'ATCG']
    dic += [a + b for a in 'ATCG' for b in 'ATCG']
    dic += [a + '0' for a in 'ATCG']
    return dic

def transformer_index_to_ATCGseq(data):

    max_indices = torch.argmax(data, dim=0)
    max_indices = max_indices.to('cpu').numpy()

    mapping = {0: "A", 1: "T", 2: "C", 3: "G"}
    sequence = [mapping[i] for i in max_indices]

    sequence_str = ''.join(sequence)
    return sequence_str
    
def trans_output_to_input(fake_im):

    sample_seq = []
    for num_sample in range(fake_im.shape[0]):
        sample_one = fake_im[num_sample,0,:,:]
        sample_seq.append(transformer_index_to_ATCGseq(sample_one))
    # print('sample_seq = ',sample_seq) 
    # pdb.set_trace()
    sample_result = []
    for seq in sample_seq:
        sample_result.append(Dimer_split_seqs(seq))
        
    sample_result = np.array(sample_result)
    sample_result = np.expand_dims(sample_result, axis=1)
    tensor = torch.from_numpy(sample_result)
    fake_img = tensor.to('cuda')
    return fake_img


def Dimer_split_seqs(seq):
    t = text_build_vocab()
    # print('t = ', t)
    # pdb.set_trace()
    ori_result = []
    dim_result = []
    pos_result = []
    
    result = ''

    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()
        ori_result.append(t.index(seq[i].upper()))

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()
        dim_result.append(t.index(seq[i:i + wt].upper()))

    pos_result += [i for i in range(1, lens + 1)]

    if ori_result[0] < 0:
        pdb.set_trace()
        print('seq = ', seq)
    
    seq_r = []
    seq_r.append(ori_result)
    seq_r.append(dim_result)
    seq_r.append(pos_result)

    return seq_r
def plot_test_prediction_result(output,label,epoch):
    val_pre = output.detach().cpu().numpy()
    val_pra = label.detach().cpu().numpy()
    
    plt.close()
    plt.figure()
    plt.plot(val_pre,label = 'val_pre')
    plt.plot(val_pra,label = 'val_pra')
    plt.legend()
    plt.title('prediction value and practice value')
    plt.savefig(f'result/epoch={epoch}')
    plt.show()
    
def compute_correlation_coefficient(output,label):
    target = output.detach().cpu().numpy()
    prediction = label.detach().cpu().numpy()

    has_nan = np.isnan(prediction).any() or np.isnan(target).any()

    if has_nan:
        print("There are NaN values in the array.")

    if np.std(prediction) == 0:
        print('The prediction results show no fluctuation.')
        return 0
    
    if np.std(target) == 0:
        print('The real data shows no fluctuation.')
        return 0

    mean_target = np.mean(target)
    mean_prediction = np.mean(prediction)

    covariance = np.mean((target - mean_target) * (prediction - mean_prediction))
    std_target = np.std(target)
    std_prediction = np.std(prediction)

    pearson_coefficient = covariance / (std_target * std_prediction)

    return pearson_coefficient
