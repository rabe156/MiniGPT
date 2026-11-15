import torch 
from torch import nn 
from .transformer import GptBlock
from torch.nn import functional as F
from .dataset import get_batch 
import os

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
        nn.init.normal_(
            self.transformer.Embeddings.weight,
            mean=0.0,
            std=0.02
        )

        nn.init.normal_(
            self.transformer.PositionalEmbeddings.weight,
            mean=0.0,
            std=0.02
        )

    def forward(self ,tokens , targets=None) :
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
        loss=None
        if targets is not None :
            FlatLogits=logits.reshape(-1,logits.size(-1))
            FlatTargets=targets.reshape(-1)
            loss=F.cross_entropy(FlatLogits,FlatTargets)

        return logits , loss  

    def train_step(model, optimizer, x, y ):
        model.train()
        # TODO 1: Clear gradients from the previous step
        optimizer.zero_grad(set_to_none=True)
        
        # TODO 2: Execute the forward pass
        logits , loss =model(x,y)
        # Your model returns (logits, loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        # TODO 3: Backpropagate the loss
        optimizer.step()
        # TODO 4: Update the parameters using AdamW

        return loss.item()
    
    def train_model(
        model,
        optimizer,
        train_data,
        val_data,
        eval_interval,
        eval_steps,
        schedular,
        batch_size=16,
        block_size=128,
        max_steps=100,
        start_step=0,best_valloss=float("inf")
    ):

        losses = []
        val_losses=[]
        device=next(model.parameters()).device
        # TODO 1: Repeat max_steps times
        for i in range (start_step,max_steps) :
            x,y=get_batch(train_data,batch_size,block_size)
            x=x.to(device)
            y=y.to(device)
            loss=model.train_step(optimizer , x, y)
            schedular.step()
            losses.append(loss)
            if(i+1) % 10 == 0 :
                print(f"train Loss is {loss}")
            if (i+1) % eval_interval == 0 :
                val_loss=model.estimate_loss(val_data,batch_size,block_size,eval_steps)
                if val_loss < best_valloss :
                    best_valloss=val_loss
                    save_checkpoint(model,optimizer,schedular,i+1,best_valloss)
                print(f"validation loss is {val_loss}")
                val_losses.append(val_loss)
        return losses  , val_losses
    
    def estimate_loss(
        model,
        val_data,
        batch_size=16,
        block_size=128,
        eval_steps=20
    ):
        device = next(model.parameters()).device
        losses = []

        # TODO 1: Switch the model to evaluation mode
        model.eval()
        # TODO 2: Disable gradient tracking
        # Hint: use a context manager
        with torch.inference_mode() :
            for steps in range(eval_steps) : 
                x,y=get_batch(val_data , batch_size , block_size)
                x=x.to(device)
                y=y.to(device)
                logits,loss=model(x,y)
                losses.append(loss.item())
        model.train()

        return sum(losses)/len(losses)


def Scheduler(optimizer ,warmupiter , maxiter):

    assert 0<warmupiter<maxiter
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmupiter
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=maxiter-warmupiter,
        eta_min=3e-5
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmupiter]
    )

    return scheduler

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step,
    val_loss,
    path="./checkpoints"
):
    checkpoint = {
        "model_weights" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scheduler_state" : scheduler.state_dict(),
        "step" : step,
        "validation_loss" : val_loss , 
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None)
    }

    # TODO 2:
    os.makedirs(path,exist_ok=True)
    checkpoint_path=os.path.join(path , f"best_model.pt")
    torch.save(checkpoint , checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_path
):
   
    checkpoint=torch.load(checkpoint_path ,map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model_weights"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])

    scheduler.load_state_dict(checkpoint["scheduler_state"])

    step=checkpoint["step"]
    val_loss=checkpoint["validation_loss"]
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    if torch.cuda.is_available():
        cuda_states = checkpoint["cuda_rng_states"]

        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return step, val_loss



