import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import torch
from utils import *
from net import predict_transformerv2
from initialize import initialize_weights
from torch.utils.data import DataLoader,Dataset

import numpy as np
import pdb
import pickle
import os
from sklearn.model_selection import KFold

# class Seq_sample():
#     def __init__(self, label, feature):
#         self.label = label
#         self.feature = feature

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def make_dataset_for_prediction_prometer(promoter, strength, seq_length=50 ):

    # folder_path = 'dataset_seq_length={0}/'.format(seq_length)

    # if not os.path.exists(folder_path):

    #     os.makedirs(folder_path)
    #     print(f"The folder '{folder_path}' has been created.")

    # else:
    #     print(f"The folder '{folder_path}' already exists.")

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
    features = []
    labels = []

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
        labels.append(label)
        features.append(feature)
        
        # sample = Seq_sample(feature = feature, label = label)

        # with open('./dataset_seq_length={0}/sample_{2}.pkl'.format(seq_length, number), 'wb') as file:
        #     pickle.dump(sample, file)

        number += 1
    # print('number = ',number)
    return features, labels


def train(params, features_array, labels_array):
    
    print('params = ',params)
    test_pearson_kfold = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(features_array)):
        print(f"Fold {fold + 1}/{k_folds}")

        print('size of train datset is: ', len(train_indices))
        print('size of test datset is: ', len(val_indices))

        train_dataset = CustomDataset(features_array[train_indices], labels_array[train_indices])
        test_dataset = CustomDataset(features_array[val_indices], labels_array[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params['train_batch_size'], shuffle=False)

        print('start compose simple gan model')
        gen = predict_transformerv2.Predict_transformer(params=params).to(device)

        initialize_weights(gen)
        print('successful compose simple gan model')

        opt_gen = torch.optim.Adam(gen.parameters(), lr=params['train_base_learning_rate'], weight_decay=params['l2_regularization'])

        loss_fc = torch.nn.MSELoss()

        loss_train =[]
        loss_test = []
        metric = []
        
        # Early stopping variables
        best_test_loss = float('inf')
        epochs_since_improvement = 0

        for epoch in range(params['train_epochs_num']):
            gen.train()
            

            if epoch > 0 and epoch % 100 == 0:

                for param_group in opt_gen.param_groups:
                    param_group['lr'] = param_group['lr'] / 5.0

            loss_train_one_epoch = 0
            loss_test_one_epoch = 0

            loss_mse = 0
            loss_pier = 0
            
            for data,target in train_loader:
                
                data = data.to(device)
                target = target.to(device)

                output = gen(data)
                
                output = torch.squeeze(output, dim=1)

                loss_gen = loss_fc(target.float(), output.float())
                loss_pi = loss_pierxun(target=target.float(),output=output.float())

                loss_gen =loss_gen.float()
                loss_pi =loss_pi.float()

                loss_all = loss_gen

                opt_gen.zero_grad()
                loss_all.backward()
                opt_gen.step()

                loss_train_one_epoch += loss_all.item()
                loss_mse += loss_gen.item()
                loss_pier += loss_pi.item()
                
            loss_train.append(loss_train_one_epoch)

            if epoch % 10 == 0:

                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Train loss: {loss_train_one_epoch/len(train_loader):.6f}****MSE loss: {loss_mse/len(train_loader):.6f}****Pierxun loss: {loss_pier/len(train_loader):.6f}"
                        )

            gen.eval()
            targets = []
            outputs = []

            for data,target in test_loader:
                
                data = data.to(device)
                target = target.to(device)

                output = gen(data)
                output = torch.squeeze(output, dim=1)
                loss_gen = loss_fc(target, output)

                targets.append(target)
                outputs.append(output)

                loss_test_one_epoch += loss_gen.detach().cpu().numpy() 

            correlation_coefficient = compute_correlation_coefficient(torch.cat(targets, dim=0), torch.cat(outputs, dim=0))
            loss_test.append(loss_test_one_epoch)

            # Early stopping logic
            if loss_test_one_epoch < best_test_loss:
                
                best_test_loss = loss_test_one_epoch
                epochs_since_improvement = 0  # Reset the counter when improvement occurs
                print('Best model saved')
                torch.save(gen,'./models/kfold_predict_{0}_mertric={1}.pth'.format(epoch,correlation_coefficient))

            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= args.early_stopping:
                print(f"Early stopping triggered after {iteration} iterations")
                break

            if epoch % 10 == 0:
                
                print(
                        f"Epoch[{epoch}/{params['train_epochs_num']}] ****Test loss: {loss_test_one_epoch/len(test_loader):.6f}********test correlation_coefficient:{correlation_coefficient}"
                        )
            
            metric.append(correlation_coefficient)


            if correlation_coefficient > 0.29:
                torch.save(gen,'./models/kfold_predict_{0}_mertric={1}.pth'.format(epoch,correlation_coefficient))
        

        dict2 = {'correlation_coefficient':max(metric),
                'dataset': sample_folder,
                'kfold':fold+1}
        
        write_good_record(dict1=params,dict2=dict2,file_path='good_record_kfold.txt')
        test_pearson_kfold.append(max(metric))

    return -max(test_pearson_kfold)

if __name__ == '__main__':
    # Early stopping parameter
    early_stopping=10
    
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)
    
    params = {
    'device_num': 4, 
 
    'dropout_rate1': 0.3254948178441311,
    'dropout_rate2': 0.36751719371886576, 

    'dropout_rate_fc': 0.4458100938040957, 

    'embedding_dim1': 64,
    'embedding_dim2': 64,

    'fc_hidden1': 210,
    'fc_hidden2': 37,

    'hidden_dim1': 128,
    'hidden_dim2': 1024,

    'l2_regularization': 1e-5,
      
    'latent_dim1': 64,
    'latent_dim2': 256,

    'num_head1': 8,
    'num_head2': 16,

    'seq_len': 50,
    'train_base_learning_rate': 0.0001,
    'train_batch_size': 512,
    'train_epochs_num': 500,

    'transformer_num_layers1': 3,
    'transformer_num_layers2': 3,
       }


    device = torch.device(f'cuda:{params["device_num"]}' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    space = {

        'train_batch_size':hp.choice('train_batch_size',[1024]),
        'seq_len':hp.choice('seq_len',[50]),
        'device_num':hp.choice('device_num',[4]),
        'train_epochs_num':hp.choice('train_epochs_num',[500]),

        'train_base_learning_rate': hp.loguniform('train_base_learning_rate', -7, -4),

        'dropout_rate1': hp.uniform('dropout_rate1', 0.1, 0.5),
        'dropout_rate2': hp.uniform('dropout_rate2', 0.1, 0.5),
        'dropout_rate_fc': hp.uniform('dropout_rate_fc', 0.1, 0.5),

        'transformer_num_layers1': hp.randint('transformer_num_layers1',1, 12),
        'transformer_num_layers2': hp.randint('transformer_num_layers2',1, 12),
        
        'l2_regularization': hp.choice('l2_regularization', [1e-4,5e-5,2e-5,5e-6,1e-6]),

        'num_head1': hp.choice('num_head1', [2, 4, 8, 16]),
        'num_head2': hp.choice('num_head2', [2, 4, 8, 16]),

        'hidden_dim1': hp.choice('hidden_dim1',[64,128,256,512,1024]),
        'latent_dim1': hp.choice('latent_dim1', [64,128, 256,512]),
        'embedding_dim1': hp.choice('embedding_dim1',[64,128, 256,512]),

        'hidden_dim2': hp.choice('hidden_dim2',[128,256,512,1024]),
        'latent_dim2': hp.choice('latent_dim2', [64, 128, 256,512]),
        'embedding_dim2': hp.choice('embedding_dim2',[64, 128, 256,512]),

        'fc_hidden1': hp.randint('fc_hidden1',64, 256),
        'fc_hidden2': hp.randint('fc_hidden2',16, 64)
    }

    # load data
    promoter = np.load('../data/promoter.npy')
    strength = np.load('../data/gene_expression.npy')

    print('strength.shape = ', strength.shape)
    print('promoter.shape = ', promoter.shape)

    features, labels = make_dataset_for_prediction_prometer(promoter, strength, seq_length=50 )

    features_array = np.array(features)
    labels_array = np.array(labels)

    train(params, features_array, labels_array)
    trials = Trials()

    objective = lambda params: train(params, features_array, labels_array)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=1000, trials=trials)

    print('best parameters:', best)
