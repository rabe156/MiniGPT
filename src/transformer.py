import torch 
from torch import nn
from attention import CausalSelfAttention

class FFN(nn.Module) :
    def __init__(self , d_model,d_ff):
        super().__init__()
        self.layer1=nn.Linear(d_model,d_ff)
        self.gelu=nn.GELU()
        self.layer2=nn.Linear(d_ff,d_model)

    def forward(self , x):
        return self.layer2(self.gelu(self.layer1(x)))


class GptBlock(nn.Module) : 
    def __init__(self,d_model=128,num_heads=4,d_ff=512) :
        super().__init__() 
        self.layerNorm1=nn.LayerNorm(d_model)
        self.attention=CausalSelfAttention(d_model,num_heads)
        self.layerNorm2=nn.LayerNorm(d_model)
        self.FFN=FFN(d_model,d_ff)

    def forward(self , x):
        x=x+self.attention(self.layerNorm1(x))
        x=x+self.FFN(self.layerNorm2(x))
        return x

