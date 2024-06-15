import pandas as pd
import numpy as np
from utils import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class Seq_sample():
    def __init__(self, label, feature):
        self.label = label
        self.feature = feature

def make_dataset_for_prediction_prometer(promoter, strength, seq_length=50 ):

    folder_path = 'dataset_seq_length={0}/'.format(seq_length)

    if not os.path.exists(folder_path):

        os.makedirs(folder_path)
        print(f"The folder '{folder_path}' has been created.")

    else:
        print(f"The folder '{folder_path}' already exists.")

    strength = np.array(strength)

    length = len(promoter[0])
    print(strength[0])
    print('length = ',length)

    strength_float = strength.astype(float)
    strength_float = np.log10(strength_float)

    strength_max = np.max(strength_float)
    strength_min = np.min(strength_float)

    print('max_strength = ',strength_max)
    print('min_strength = ',strength_min)

    number = 0

    for sequence, score in zip(promoter, strength_float):

        if  50 != len(sequence) or score == 0.0:

            print('the length of sequence is ',len(sequence))
            print('promoter = ',sequence.upper())
            print('strength = ',score)
            continue

        feature = Dimer_split_seqs(sequence[-seq_length:]) 
        feature = np.array(feature)
        feature = feature.astype(int)

        label = (score -  strength_min)/( strength_max -   strength_min)
        sample = Seq_sample(feature = feature, label = label)

        with open('./dataset_seq_length={0}/sample_{2}.pkl'.format(seq_length, number), 'wb') as file:
            pickle.dump(sample, file)

        number += 1
    print('number = ',number)

if __name__ == "__main__":

    promoter = np.load('../data/promoter.npy')
    strength = np.load('../data/gene_expression.npy')

    print('strength.shape = ', strength.shape)
    print('promoter.shape = ', promoter.shape)

    make_dataset_for_prediction_prometer(promoter, strength, seq_length=50 )



