import pdb
import sys

import torch
import torchvision


def check_weights(weight_path: str):
    """
    Use this method to begin exploring saved checkpoints
    """

    try:
        checkpoint = torch.load(weight_path)
    except:
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))

    checkpoint_type = type(checkpoint)
    print(f"{checkpoint_type = }")
    if isinstance(checkpoint, dict):
        print(f"{checkpoint.keys()}")

    # Check compatabilityw with ResNet50
    resnet50 = torchvision.models.resnet50()
    print(f"{resnet50.state_dict().keys() = }")
    resnet50.load_state_dict(checkpoint, strict=True)

    pdb.set_trace()



if __name__ == "__main__":
    check_weights(*sys.argv[1:])
