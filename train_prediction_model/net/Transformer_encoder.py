import torch
import torch.nn as nn
from torch.nn import functional as F
from net.transformer import PositionalEncoding, TransformerEncoder

class Predict_encoder(nn.Module):
    def __init__(self,nhead,layers,hidden_dim,latent_dim,embedding_dim,seq_len,probs,device='cuda'):
        super(Predict_encoder,self).__init__()
        
        self.layers = layers # encoder的数目
        self.embedding_dim = embedding_dim #嵌入的维度
        self.seq_len = seq_len # 序列的长度
        self.nhead = nhead # 注意力头数
        self.hidden_dim = hidden_dim #encoder中全连接层的隐藏层神经元数目
        self.probs = probs # encoder中全连接层的dropout系数
        self.device = device
        self.latent_dim = latent_dim # 将编码器得到的信息最终以想要的维度输出，不接这个那么输出的维度一直是embedding dim
        self.src_mask = None
        
        self.pos_encoder = PositionalEncoding(
           device=self.device,d_model=self.embedding_dim, max_len=self.seq_len
        )
        
        self.transformer_encoder = TransformerEncoder(
            num_layers=self.layers,
            input_dim=self.embedding_dim,
            num_heads=self.nhead,
            dim_feedforward=self.hidden_dim,
            dropout=self.probs,
        )
        
        self.glob_attn_module = nn.Sequential(
                nn.Linear(self.embedding_dim, 1), nn.Softmax(dim=1)
            )
        
        self.fc1 = nn.Linear(self.embedding_dim, self.latent_dim)
    
    def _generate_square_subsequent_mask(self, sz):
        """create mask for transformer
        Args:
            sz (int): sequence length
        Returns:
            torch tensor : returns upper tri tensor of shape (S x S )
        """
        mask = torch.ones((sz, sz), device=self.device)
        mask = (torch.triu(mask) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def transformer_encoding(self, embedded_batch):
        """transformer logic
        Args:
            embedded_batch ([type]): output of nn.Embedding layer (B x S X E)
        mask shape: S x S
        """
        if self.src_mask is None or self.src_mask.size(0) != len(embedded_batch):
            
            self.src_mask = self._generate_square_subsequent_mask(
                embedded_batch.size(1)
            )
            
        # self.embed gives output (batch_size,sequence_length,num_features)
        pos_encoded_batch = self.pos_encoder(embedded_batch)

        # TransformerEncoder takes input (sequence_length,batch_size,num_features)
        output_embed = self.transformer_encoder(pos_encoded_batch, self.src_mask)
        return output_embed
    
    def encoder(self,embedded_batch):
        
        # 输入嵌入后的信息
        output_embed = self.transformer_encoding(embedded_batch)

        glob_attn = self.glob_attn_module(output_embed)  # output should be B x S x 1
        
        z_rep = torch.bmm(glob_attn.transpose(-1, 1), output_embed).squeeze()

        # to regain the batch dimension
        if len(embedded_batch) == 1:
            z_rep = z_rep.unsqueeze(0)

        z_rep = self.fc1(z_rep)

        return z_rep
    
    def forward(self,input):
        return self.encoder(input)
    
if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    noise_dim = 1
    in_channels = 1
    feature_g = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('*********dedvice =',device,'*********')
    Pre_model = Predict_encoder(nhead = 4,layers=4,hidden_dim=4,latent_dim=16,embedding_dim=100,seq_len=100,probs=0.1,device='cuda')
    Pre_model = Pre_model.to(device)
    z = torch.randn(size = (64,100,100),device='cuda')
    # summary(gen,input_size=(1,noise_dim,1,1))
    print(Pre_model(z).shape)