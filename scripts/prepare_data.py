from datasets import load_dataset
import torch

def LoadData(DatasetID) :
    TrainData=load_dataset(
        DatasetID,
        split="train",
        streaming=True
    )
    ValData=load_dataset(
         DatasetID,
         split="train",
         streaming=True
    )
    TrainStories=[story for story in TrainData.take(9000)]
    ValStories=[story for story in ValData.take(9000)]
    print(len(TrainStories))



DATASET_ID = "skeskinen/TinyStories-hf"
LoadData(DATASET_ID)