import torch 
from torch import nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self , d_model=128 , num_heads=4 ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model=d_model
        self.num_heads=num_heads
        self.projections=nn.Linear(d_model,3*d_model)
        self.OutputProj=nn.LazyLinear(d_model)


    def forward(self,x) :
        B ,SL ,C=x.size()
        Projections=self.projections(x)
        Q,K,V=Projections.split(self.d_model , 2)

        Q=Q.reshape(B,SL,self.num_heads , -1).transpose(1,2)
        K=K.reshape(B,SL,self.num_heads , -1).transpose(1,2)
        V=V.reshape(B,SL,self.num_heads , -1).transpose(1,2)

        AttentionOutput=F.scaled_dot_product_attention(Q,K,V , is_causal=True,dropout_p=0.0)

        AttentionOutput=AttentionOutput.transpose(1,2)
        AttentionOutput=AttentionOutput.reshape(B , SL , -1)

        Attention=self.OutputProj(AttentionOutput)
        return Attention

    