from datasets import load_dataset
import torch
import tiktoken

def LoadData(DatasetID) :
    TrainData=load_dataset(
        DatasetID,
        split="train",
        streaming=True
    )
    ValData=load_dataset(
         DatasetID,
         split="validation",
         streaming=True
    )
    TrainStories=[story["text"] for story in TrainData.take(9000)]
    ValStories=[story["text"] for story in ValData.take(1000)]
    return TrainStories , ValStories 

def tokenize(Stories) :
    Tokenizer=tiktoken.get_encoding("gpt2")
    Tokens=[]
    for story in Stories :
        Token=Tokenizer.encode(story)
        Token.append(Tokenizer.eot_token)
        Tokens.extend(Token)
    Tokens=torch.tensor(Tokens , dtype=torch.long)
    return Tokens
