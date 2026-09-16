from resnet import ResNet18

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Path to the dataset folder (dataset will be downloaded here)",
    )
    return parser.parse_args()

args = parse_args()

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True)