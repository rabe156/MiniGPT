import torch 
def get_batch(data, batch_size, block_size):

    # TODO 1:
    # Generate batch_size random starting indices.
    indices = torch.randint(
    low=0,
    high=len(data) - block_size,
    size=(batch_size,)
    )
    XSequence=[]
    YSequence=[]
    # Ensure there is room for block_size + 1 tokens.
    assert len(data) > block_size
    for i in indices :
    # TODO 2:
        i=i.item()
        x = data[i : i + block_size]
        y = data[i + 1 : i + block_size + 1]
    # TODO 3:
    # Stack the input sequences into one tensor.
        XSequence.append(x)
        YSequence.append(y)    
    # TODO 4:
    # Stack the target sequences into one tensor.
    x=torch.stack(XSequence)
    y=torch.stack(YSequence)
    return x, y