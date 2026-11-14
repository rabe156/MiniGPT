import torch 
from torch import nn 
from transformer import GptBlock

class MiniGpt (nn.Module) :
    def __init__(self,vocab_size ,d_model,context_size ,num_heads,d_ff,num_layers) :
        super().__init__()
        self.transformer=nn.ModuleDict(dict(
            Embeddings=nn.Embedding(vocab_size , d_model),
            PositionalEmbeddings=nn.Embedding(context_size,d_model),
            Blocks=nn.ModuleList([GptBlock(d_model ,num_heads ,d_ff) for _ in range(num_layers)]),
            LayerNorm=nn.LayerNorm(d_model),
        ))
        self.modelhead=nn.Linear(d_model,vocab_size,bias=False)
        self.modelhead.weight = self.transformer.Embeddings.weight
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.context_size=context_size
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.num_layers=num_layers

    def forward(self ,tokens) :
        Batch,Context=tokens.size()
        assert self.context_size >= Context ,f"max context size={self.context_size} ,cuurent sequence lenght is {Context}"
        Positions=torch.arange(0,Context,dtype=torch.long,device=tokens.device)
        PositionsEmbedding=self.transformer.PositionalEmbeddings(Positions)
        Embeddings=self.transformer.Embeddings(tokens)
        x=Embeddings+PositionsEmbedding
        for block in self.transformer.Blocks :
            x=block(x)
        x=self.transformer.LayerNorm(x)
        logits=self.modelhead(x)

        return logits   