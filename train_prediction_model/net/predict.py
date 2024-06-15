"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/2 12:08
"""
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers,num_classes):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 可以使用不同的汇聚策略
        x = self.fc(x)
        return x
# 定义一维卷积残差模块
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels

        # 定义两个卷积层
        self.conv1 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=27, padding=13)
        self.conv2 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=27, padding=13)
        
        self.batch_norm = nn.BatchNorm1d(self.num_channels)
    

    def forward(self, x):
        res = x
        for _ in range(2):
            res = self.batch_norm(res)
            res = F.relu(res)
            # print('res.shape = ',res.shape)
            # print('bias1.shape = ',self.bias1.shape)
            
            # res = self.conv1(res) + self.bias1
            res = self.conv1(res)
            res = self.batch_norm(res)
            res = F.relu(res)
            # res = self.conv2(res) + self.bias2
            res = self.conv2(res)
            
        return x + (0.3 * res)

class predict(torch.nn.Module):
    def __init__(self,params):
        super(predict, self).__init__()
        self.dropout_rate = params['dropout_rate']
        
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        
        self.conv1_ori = nn.Conv1d(100, params['conv1d_filters_num'], kernel_size=params['conv1d_filters_size'], padding='same')
        self.conv1_dim = nn.Conv1d(100, params['conv1d_filters_num'], kernel_size=params['conv1d_filters_size'], padding='same')
        self.conv1_pos = nn.Conv1d(100, params['conv1d_filters_num'], kernel_size=params['conv1d_filters_size'], padding='same')
        
        # self.conv1_ori_res = nn.ModuleList([ResidualBlock(params['conv1d_filters_num']) for _ in range(50)])
        # self.conv1_dim_res = nn.ModuleList([ResidualBlock(params['conv1d_filters_num']) for _ in range(50)])
        # self.conv1_pos_res = nn.ModuleList([ResidualBlock(params['conv1d_filters_num']) for _ in range(50)])
        
        self.pool1_ori = nn.AvgPool1d(1)
        self.pool1_dim = nn.AvgPool1d(1)
        self.pool1_pos = nn.AvgPool1d(1)
        
        self.conv2 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=params['conv1d_filters_size'])
        self.conv3 = nn.Conv1d(params['conv1d_filters_num'], params['conv1d_filters_num'], kernel_size=params['conv1d_filters_size'])

        self.conv2_res = nn.ModuleList([ResidualBlock(512) for _ in range(20)])
        self.conv3_res = nn.ModuleList([ResidualBlock(512) for _ in range(20)])
        
        
        # Define MLP
        self.mlp = MLP(input_dim=62464, output_dim= 512, hidden_layer_num= 40, hidden_layer_units_num= 512,dropout=0.03)     

        # dropout层
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.fc_bn_1 = nn.BatchNorm1d(128)
        self.fc_bn_2 = nn.BatchNorm1d(32)
        self.fc_bn_3 = nn.BatchNorm1d(8)
        
        self.final_fc_1 = nn.Linear(512,128)
        self.final_fc_2 = nn.Linear(128,32)
        self.final_fc_3 = nn.Linear(32,8)
        self.final_fc_4 = nn.Linear(8,1)
        
        
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
        input_pos = x[:, 2, :]
        # print('input_ori.shape = ',input_ori.shape)
        # print('input_dim.shape = ',input_dim.shape)
        # print('input_pos.shape = ',input_pos.shape)
        # print('input_ori = ',input_ori)
        
        embeded_ori = self.embedding_ori(input_ori)
        embeded_dim = self.embedding_dim(input_dim)
        embeded_pos = self.embedding_pos(input_pos)
        
        # print('embeded_ori = ',embeded_ori[0])
        # print('embeded_dim = ',embeded_dim[0])
        # print('embeded_pos = ',embeded_pos[0])
        
        # print('embeded_ori.shape = ',embeded_ori.shape)
        # print('embeded_dim.shape = ',embeded_dim.shape)
        # print('embeded_pos.shape = ',embeded_pos.shape)

        
        conv_ori = self.conv1_ori(embeded_ori)
        conv_dim = self.conv1_dim(embeded_dim)
        conv_pos = self.conv1_pos(embeded_pos)
        
        conv_ori = nn.ReLU()(conv_ori)
        conv_dim = nn.ReLU()(conv_dim)
        conv_pos = nn.ReLU()(conv_pos)
        
        pool_ori = self.pool1_ori(conv_ori)
        pool_dim = self.pool1_dim(conv_dim)
        pool_pos = self.pool1_pos(conv_pos)
        
        # for res_block in self.conv1_ori_res:
        #     outputs = res_block(outputs)
        #     # print('outputs3.shape = ',outputs.shape)
        
        # for res_block in self.conv1_dim_res:
        #     outputs = res_block(outputs)
        #     # print('outputs3.shape = ',outputs.shape)
        
        # for res_block in self.conv1_pos_res:
        #     outputs = res_block(outputs)
        #     # print('outputs3.shape = ',outputs.shape)
        
        
        drop_ori = self.dropout(pool_ori)
        drop_dim = self.dropout(pool_dim)
        drop_pos = self.dropout(pool_pos)
        
        
        pool_seq = torch.add(pool_dim,pool_ori)
        drop_seq = torch.add(drop_dim,drop_ori)

        pool1 = torch.add(pool_seq, pool_pos)
        drop1 = torch.add(drop_seq, drop_pos)
        
        conv2 = self.conv2(pool1)
        conv3 = self.conv3(drop1)
        
        # 加入残差模块
        for res_block in self.conv2_res:
            conv2 = res_block(conv2)
            # print('outputs3.shape = ',outputs.shape)
        for res_block in self.conv3_res:
            conv3 = res_block(conv3)
        
        conv = conv2 + conv3

        conv = conv.view(conv.shape[0], -1)
        # print('conv.shape = ',conv.shape)
        output = self.mlp(conv)
        
        output = self.final_fc_1(output)
        output = self.fc_bn_1(output)
        output = nn.ReLU()(output)
        
        output = self.final_fc_2(output)
        output = self.fc_bn_2(output)
        output = nn.ReLU()(output)
        
        output = self.final_fc_3(output)
        output = self.fc_bn_3(output)
        output = nn.ReLU()(output)
        
        output = self.final_fc_4(output)
        # print('output.shape', output.shape)
        # pdb.set_trace()
        return nn.Sigmoid()(output)
        
        
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
    in_channles = 3
    H,W = 64,64
    x = torch.randn(size = (1,in_channles,H,W))
    disc = predict()
    # summary(disc, input_size=(1, in_channles, H,W))
    print(disc(x).shape)