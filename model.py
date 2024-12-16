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
    def __init__(
