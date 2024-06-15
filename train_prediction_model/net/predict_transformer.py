import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn



class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, hidden_dim, num_layers,num_classes):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.Transformer(
            d_model=128,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x,pos):
        x = self.transformer(x,pos)
        x = x.mean(dim=1)  # 可以使用不同的汇聚策略
        x = self.fc(x)
        return x

class Predict_transformer(torch.nn.Module):
    def __init__(self,params,num_heads=8, hidden_dim=256, num_layers=6,num_classes=64):
        super(Predict_transformer, self).__init__()
        self.dropout_rate = params['dropout_rate']
        
        self.trans_ori_pos = TransformerEncoder(num_heads, hidden_dim, num_layers, num_classes)
        self.trans_dim_pos = TransformerEncoder(num_heads, hidden_dim, num_layers, num_classes)
        
        # Define the layers as PyTorch modules
        self.embedding_ori = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        self.embedding_dim = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        self.embedding_pos = torch.nn.Embedding(128, params['nuc_embedding_outputdim'])
        
        # Define MLP
        self.mlp = MLP(input_dim=2*num_classes, output_dim= 32, hidden_layer_num= 10, hidden_layer_units_num= 64,dropout=0.1)     

        # dropout层
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.final_fc = nn.Linear(32,1)
        
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
        
        # ori_pos = embeded_ori + embeded_pos
        # dim_pos = embeded_dim + embeded_pos
        
        ori_pos = self.trans_ori_pos(embeded_ori , embeded_pos)
        dim_pos = self.trans_dim_pos(embeded_dim , embeded_pos)
        
        ori_dim_pos = torch.cat((ori_pos, dim_pos), dim=-1)
        output = self.mlp(ori_dim_pos)
        
        output = self.final_fc(output)
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
    disc = Predict_transformer()
    # summary(disc, input_size=(1, in_channles, H,W))
    print(disc(x).shape)