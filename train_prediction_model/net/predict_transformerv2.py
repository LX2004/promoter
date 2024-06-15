import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn
from net.Transformer_encoder import Predict_encoder

class ResidualBlock(nn.Module):
    def __init__(self, num_channels,kernel_size,padding):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels

        # 定义两个卷积层
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(num_channels) 

        self.ac = nn.LeakyReLU()
        
    def forward(self, x):
        res = x
        for _ in range(2):
            res = self.conv1(res)
            res = self.batch_norm(res) 
            # res = F.relu(res)
            res = self.ac(res)

            res = self.conv2(res)
            res = self.batch_norm(res)
            # res = F.relu(res)
            
        return x + res

class Predict_transformer(torch.nn.Module):
    def __init__(self,params):
        super(Predict_transformer, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')
        # self.trans_all = Predict_encoder(nhead = 4,layers=4,hidden_dim=4,latent_dim=64,embedding_dim=100,seq_len=100,probs=0.1,device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        # self.embedding_pos = torch.nn.Embedding(100, params['embedding_dim'])
        
        # Define 1D-CNN
        # self.cnn_ori = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_all = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_ori_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])

        
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'],params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
        
    def forward(self, X):
        # print('X_in.ori = ', X[1,0,0,:])
        # print('X_in.dim = ', X[1,0,1,:])
        # print('X_in.pos = ', X[1,0,2,:])
        x = X.to(torch.int)
        # Split input X: [bt,1,3,100]
        # x = X[:,0,:,:]
        # print('x = ',x[1])
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        # embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        
        # embeded_dim = self.cnn_dim(embeded_dim)
        # embeded_ori = self.cnn_ori(embeded_ori)
        # embeded_ori_dim = self.cnn_ori_dim( embeded_ori +  embeded_dim )
        
        # cnn_all = self.cnn_all(embeded_ori_dim + embeded_pos)
      
        # print('embeded_ori.shape = ', embeded_ori.shape)
        
        # print('start transformer encoder')
        # all_trans = self.trans_all(embeded_ori_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos), dim=-1)
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        # output = self.dropout(output)

        output = self.final_fc3(output)
        
        # output = self.relu(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return self.relu(output)


class transformer_ont_biofeat_classification(torch.nn.Module):
    '''
    参考论文：《TransCrispr: Transformer Based Hybrid Model for Predicting CRISPR/Cas9 Single Guide RNA Cleavage Efficiency》
    github连接为：https://github.com/BioinfoApollo/TransCrispr/blob/main/BioNet.py
    '''
    def __init__(self,params):
        super(transformer_ont_biofeat_classification, self).__init__()

        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['nuc_embedding_outputdim'],seq_len=params['conv1d_filters_num'],probs=params['dropout_rate'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['nuc_embedding_outputdim'],seq_len=params['conv1d_filters_num'],probs=params['dropout_rate'],device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        
        # Define 1D-CNN
        self.cnov1d_ori = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.cnov1d_dim = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.cnov1d_pos = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        
        self.conv2 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.conv3 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        
        # 平均池化层
        # self.avg_pool_ori = nn.AvgPool1d(kernel_size = params['embedding_dim2'])
        # self.avg_pool_dim = nn.AvgPool1d(kernel_size = params['embedding_dim2'])

        self.ac = nn.ReLU()

        # dropout层
        self.dropout_ori = nn.Dropout(p=params['dropout_rate'])
        self.dropout_dim = nn.Dropout(p=params['dropout_rate'])
        self.dropout_fc = nn.Dropout(p=params['dropout_rate'])
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'] + 4 + params['nuc_embedding_outputdim'], params['fc_hidden1'])
        self.final_fcbn1 = nn.BatchNorm1d(params['fc_hidden1'])

        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fcbn2 = nn.BatchNorm1d(params['fc_hidden2'])

        self.final_fc3 = nn.Linear(params['fc_hidden2'], params['fc_hidden3'])
        self.final_fcbn3 = nn.BatchNorm1d(params['fc_hidden3'])

        self.final_fc4 = nn.Linear(params['fc_hidden3'], 5)
        
    def forward(self, x, bio):
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        input_pos = x[:, 2, :] - 1
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        conv1_nuc = self.cnov1d_ori(embeded_ori)
        conv1_dimer = self.cnov1d_dim(embeded_dim)

        # 进入激活函数
        conv1_nuc = self.ac(conv1_nuc)
        conv1_dimer = self.ac(conv1_dimer)

        # print('conv1_nuc.shape = ', conv1_nuc.shape)
        # print('conv1_dimer.shape = ', conv1_dimer.shape)

        # 进入池化层
        # pool1_nuc = self.avg_pool_ori(conv1_nuc)
        pool1_nuc = torch.mean(conv1_nuc, dim=1)

        # pool1_dimer = self.avg_pool_dim(conv1_dimer)
        pool1_dimer = torch.mean(conv1_dimer, dim=1)

        # 进入dropout层
        drop1_dimer = self.dropout_dim(conv1_dimer)
        drop1_nuc = self.dropout_dim(conv1_nuc)

        # print('drop1_nuc.shape = ', drop1_nuc.shape)
        # print('drop1_dimer.shape = ', drop1_dimer.shape)
        # print('embeded_pos.shape = ', embeded_pos.shape)

        # 叠加位置信息
        drop1_ori = drop1_nuc + self.cnov1d_pos(embeded_pos)
        drop1_dim = drop1_dimer + self.cnov1d_pos(embeded_pos)

        # 再次进入卷积层
        conv2 = self.conv2(drop1_ori)
        conv3 = self.conv3(drop1_dim)

        # 进入激活函数
        conv2 = self.ac(conv2)
        conv3 = self.ac(conv3)

        # print('conv3.shape = ', conv3.shape)
        
        # 进入Transformer
        ori_pos = self.trans_ori_pos(conv2)
        dim_pos = self.trans_dim_pos(conv3)

        # 叠加池化信息
        output = torch.cat((0.2*(pool1_nuc + pool1_dimer), 0.8*ori_pos, 0.8*dim_pos, bio), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.final_fcbn1(output)
        # output = self.dropout_fc(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.final_fcbn2(output)
        # output = self.dropout_fc(output)

        output = self.final_fc3(output)
        output = self.ac(output)
        output = self.final_fcbn3(output)
        # output = self.dropout_fc(output)

        output = self.final_fc4(output)
        
        return output 

class transformer_ont_biofeat(torch.nn.Module):
    '''
    参考论文：《TransCrispr: Transformer Based Hybrid Model for Predicting CRISPR/Cas9 Single Guide RNA Cleavage Efficiency》
    github连接为：https://github.com/BioinfoApollo/TransCrispr/blob/main/BioNet.py
    '''
    def __init__(self,params):
        super(transformer_ont_biofeat, self).__init__()

        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['nuc_embedding_outputdim'],seq_len=params['conv1d_filters_num'],probs=0.1,device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['nuc_embedding_outputdim'],seq_len=params['conv1d_filters_num'],probs=0.1,device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(50, params['nuc_embedding_outputdim'])
        
        # Define 1D-CNN
        self.cnov1d_ori = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.cnov1d_dim = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.cnov1d_pos = nn.Conv1d(params['seq_len'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        
        self.conv2 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        self.conv3 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=2*params['conv1d_filters_size']+1, padding=params['conv1d_filters_size'])
        
        # 平均池化层
        # self.avg_pool_ori = nn.AvgPool1d(kernel_size = params['embedding_dim2'])
        # self.avg_pool_dim = nn.AvgPool1d(kernel_size = params['embedding_dim2'])

        self.ac = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # dropout层
        self.dropout_ori = nn.Dropout(p=params['dropout_rate'])
        self.dropout_dim = nn.Dropout(p=params['dropout_rate'])
        self.dropout_fc = nn.Dropout(p=params['dropout_rate'])
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'] + 4 + params['nuc_embedding_outputdim'], params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])

        self.final_fc3 = nn.Linear(params['fc_hidden2'], params['fc_hidden3'])
        self.final_fc4 = nn.Linear(params['fc_hidden3'], 1)
        
    def forward(self, x, bio):
        
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        input_pos = x[:, 2, :] - 1
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        conv1_nuc = self.cnov1d_ori(embeded_ori)
        conv1_dimer = self.cnov1d_dim(embeded_dim)

        # 进入激活函数
        conv1_nuc = self.ac(conv1_nuc)
        conv1_dimer = self.ac(conv1_dimer)

        # print('conv1_nuc.shape = ', conv1_nuc.shape)
        # print('conv1_dimer.shape = ', conv1_dimer.shape)

        # 进入池化层
        # pool1_nuc = self.avg_pool_ori(conv1_nuc)
        pool1_nuc = torch.mean(conv1_nuc, dim=1)

        # pool1_dimer = self.avg_pool_dim(conv1_dimer)
        pool1_dimer = torch.mean(conv1_dimer, dim=1)

        # 进入dropout层
        drop1_dimer = self.dropout_dim(conv1_dimer)
        drop1_nuc = self.dropout_dim(conv1_nuc)

        # print('drop1_nuc.shape = ', drop1_nuc.shape)
        # print('drop1_dimer.shape = ', drop1_dimer.shape)
        # print('embeded_pos.shape = ', embeded_pos.shape)

        # 叠加位置信息
        drop1_ori = drop1_nuc + self.cnov1d_pos(embeded_pos)
        drop1_dim = drop1_dimer + self.cnov1d_pos(embeded_pos)

        # 再次进入卷积层
        conv2 = self.conv2(drop1_ori)
        conv3 = self.conv3(drop1_dim)

        # 进入激活函数
        conv2 = self.ac(conv2)
        conv3 = self.ac(conv3)

        # print('conv3.shape = ', conv3.shape)
        
        # 进入Transformer
        ori_pos = self.trans_ori_pos(conv2)
        dim_pos = self.trans_dim_pos(conv3)

        # 叠加池化信息
        output = torch.cat((0.2*(pool1_nuc + pool1_dimer), 0.8*ori_pos, 0.8*dim_pos, bio), dim=-1)

        output = self.final_fc1(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc3(output)
        output = self.ac(output)
        output = self.dropout_fc(output)

        output = self.final_fc4(output)
        
        return self.sigmoid(output)     

class Predict_transformer_bio(torch.nn.Module):
    def __init__(self,params):
        super(Predict_transformer_bio, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')
        # self.trans_all = Predict_encoder(nhead = 4,layers=4,hidden_dim=4,latent_dim=64,embedding_dim=100,seq_len=100,probs=0.1,device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        # self.embedding_pos = torch.nn.Embedding(100, params['embedding_dim'])
        
        # Define 1D-CNN
        # self.cnn_ori = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_all = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_ori_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])

        
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'] + 4, params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'],params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'],1)
        
    def forward(self, x, bio):
        # print('X_in.ori = ', X[1,0,0,:])
        # print('X_in.dim = ', X[1,0,1,:])
        # print('X_in.pos = ', X[1,0,2,:])
        # x = X.to(torch.int)
        # Split input X: [bt,1,3,100]
        # x = X[:,0,:,:]
        # print('x = ',x[1])
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        # embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        
        # embeded_dim = self.cnn_dim(embeded_dim)
        # embeded_ori = self.cnn_ori(embeded_ori)
        # embeded_ori_dim = self.cnn_ori_dim( embeded_ori +  embeded_dim )
        
        # cnn_all = self.cnn_all(embeded_ori_dim + embeded_pos)
      
        # print('embeded_ori.shape = ', embeded_ori.shape)
        
        # print('start transformer encoder')
        # all_trans = self.trans_all(embeded_ori_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos, bio), dim=-1)
        # output = torch.cat((output, bio), dim=-1)
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        output = self.dropout(output)

        output = self.final_fc3(output)
        output = self.ac(output)
        
        # output = self.relu(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return self.relu(output)
    

class Classification_transformer(torch.nn.Module):
    def __init__(self,params):
        super(Classification_transformer, self).__init__()

        self.dropout_rate_fc = params['dropout_rate_fc']
        self.relu = nn.ReLU()

        self.trans_ori_pos = Predict_encoder(nhead = params['num_head1'],layers = params['transformer_num_layers1'],hidden_dim=params['hidden_dim1'],latent_dim=params['latent_dim1'],embedding_dim=params['embedding_dim1'],seq_len=params['seq_len'],probs=params['dropout_rate1'],device='cuda')
        self.trans_dim_pos = Predict_encoder(nhead = params['num_head2'],layers = params['transformer_num_layers2'],hidden_dim=params['hidden_dim2'],latent_dim=params['latent_dim2'],embedding_dim=params['embedding_dim2'],seq_len=params['seq_len'],probs=params['dropout_rate2'],device='cuda')
        # self.trans_all = Predict_encoder(nhead = 4,layers=4,hidden_dim=4,latent_dim=64,embedding_dim=100,seq_len=100,probs=0.1,device='cuda')
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(100, params['embedding_dim1'])
        self.embedding_dim = torch.nn.Embedding(100, params['embedding_dim2'])
        # self.embedding_pos = torch.nn.Embedding(100, params['embedding_dim'])
        
        # Define 1D-CNN
        # self.cnn_ori = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_all = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])
        # self.cnn_ori_dim = ResidualBlock(num_channels=100,kernel_size=2*params['conv1d_padding']+1,padding=params['conv1d_padding'])

        
        # dropout层
        self.ac = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate_fc)
        
        self.final_fc1 = nn.Linear(params['latent_dim1'] + params['latent_dim2'],params['fc_hidden1'])
        self.final_fc2 = nn.Linear(params['fc_hidden1'], params['fc_hidden2'])
        self.final_fc3 = nn.Linear(params['fc_hidden2'], 5)
        
    def forward(self, X):
        # print('X_in.ori = ', X[1,0,0,:])
        # print('X_in.dim = ', X[1,0,1,:])
        # print('X_in.pos = ', X[1,0,2,:])
        x = X.to(torch.int)
        # Split input X: [bt,1,3,100]
        # x = X[:,0,:,:]
        # print('x = ',x[1])
        input_ori = x[:, 0, :]
        input_dim = x[:, 1, :]
        # input_pos = x[:, 2, :] - 1
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        # print('start embedding')
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        # embeded_pos = self.embedding_pos(input_pos)
        
        # 进入卷积模块
        
        # embeded_dim = self.cnn_dim(embeded_dim)
        # embeded_ori = self.cnn_ori(embeded_ori)
        # embeded_ori_dim = self.cnn_ori_dim( embeded_ori +  embeded_dim )
        
        # cnn_all = self.cnn_all(embeded_ori_dim + embeded_pos)
      
        # print('embeded_ori.shape = ', embeded_ori.shape)
        
        # print('start transformer encoder')
        # all_trans = self.trans_all(embeded_ori_dim)
        
        ori_pos = self.trans_ori_pos(embeded_ori)
        dim_pos = self.trans_dim_pos(embeded_dim)
        # print('end transformer encoder')
        
        output = torch.cat((ori_pos, dim_pos), dim=-1)
        # output = self.mlp(ori_dim_pos)
        
        output = self.final_fc1(output)
        output = self.ac(output)
        # output = self.relu(output)
        output = self.dropout(output)

        output = self.final_fc2(output)
        output = self.ac(output)
        # output = self.dropout(output)

        output = self.final_fc3(output)
        
        # output = self.relu(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return self.relu(output)


        
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_num, hidden_layer_units_num, dropout):
        super(MLP, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_layer_units_num))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        for _ in range(hidden_layer_num - 1):
            layers.append(torch.nn.Linear(hidden_layer_units_num, hidden_layer_units_num))
            layers.append(torch.nn.BatchNorm1d(hidden_layer_units_num))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_layer_units_num, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
if __name__ == '__main__':
    params = {
    'train_batch_size':64,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00005,
    'model_save_file':'./models/BestModel_WT_withbio.h5',
    'dropout_rate':0.2,
    'nuc_embedding_outputdim':100,
    'conv1d_filters_size':7,
    'conv1d_filters_num':512,
    'transformer_num_layers':4,
    'transformer_final_fn':198,
    'transformer_ffn_1stlayer':111,
    'dense1':176,
    'dense2':88,
    'dense3':22
}
    in_channles = 3
    H,W = 64,64
    x = torch.ones(size = (64,in_channles,100), device = 'cuda')
    predict_model = Predict_transformer(params).to('cuda')
    
    # summary(disc, input_size=(1, in_channles, H,W))
    print(predict_model(x).shape)