import torch
import sys
import pdb

def check_weights(weight_path:str):
    """
    Use this method to begin exploring saved checkpoints
    """
    checkpoint = torch.load(weight_path)
    checkpoint_type = type(checkpoint)
    print(f"{checkpoint_type = }")
    if isinstance(checkpoint, dict):
        print(f"{checkpoint.keys()}")
    pdb.set_trace()

if __name__ == "__main__":
    check_weights(*sys.argv[1:])


