import numpy as np
import torch
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

ws=13
fs=(ws+1)//2
oc=9
encoder_in=64
dim_f=64
L1O=128
png_in=fs*fs

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class MyModel(torch.nn.Module):
    def __init__(self,d_model=oc*png_in,num_encoder_layers=5,nhead=8,dropout=0.1,dim_feedforward=dim_f,batch_size=32):
        super(MyModel, self).__init__()
        
        self.conv0 = torch.nn.Conv2d(in_channels=9, out_channels=oc, kernel_size=3, stride=2, padding=1) 
        self.conv1 = torch.nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.embedding_layers = torch.nn.ModuleList([Linear(png_in, encoder_in) for _ in range(oc*2*3)])

        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)#encoder_in
        encoder_norm =LayerNorm(encoder_in)#d_model
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        #self.linear0=Linear(d_model*2,encoder_in)
        #self.linear0=Linear(49,768)
        self.linear1=Linear(encoder_in*oc*6, L1O)
        self.linear2=Linear(L1O, 15)   
        self.position_embedding = position_embeddings(torch.arange(batch_size*d_model*2*3), 1)
        self.d_model=d_model

        self.dropout1 = Dropout(p=0.25) 
        self.dropout2 = Dropout(p=0.1) 

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, encoder_in))
        torch.nn.init.normal_(self.cls_token)   

    def forward(self, value):
        batch_size=value.shape[0]
        value=relu(self.conv0(value))
        value=relu(self.conv1(value))
        value2=relu(self.conv2(value))
        value3=relu(self.conv3(value2))
        value=torch.cat((value,value2,value3),1)

        del value3
        value2=torch.transpose(value, 2, 3)
        value=torch.cat((value,value2),1)
        del value2

        position_ids = torch.arange(batch_size * self.d_model*2*3, dtype=torch.long, device=value.device)  
        position_embeds = self.position_embedding(position_ids).view(batch_size, self.d_model*2*3)    
        value=torch.reshape(value,(batch_size,-1))

        value = value + position_embeds
        value=value.view(batch_size,oc*2*3,-1)

        embedded_values = []
        for i in range(oc*2*3):
            embedded_value = self.embedding_layers[i](value[:, i, :])
            embedded_values.append(embedded_value)

        value = torch.stack(embedded_values, dim=1)
        value=self.encoder(value)
        value=value.view(batch_size,-1)
        value=value.squeeze(1)

        value=relu(self.linear1(value))
        value=self.linear2(value)
        return value