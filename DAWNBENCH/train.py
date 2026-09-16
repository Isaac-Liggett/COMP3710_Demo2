"""
Code Adapted from 
https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/
"""

from resnet import ResNet18


from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import random

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 512
LR = 0.1 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Path to the dataset folder (dataset will be downloaded here)",
    )
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

class Cutout:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1, y2 = max(0, y - self.size // 2), min(h, y + self.size // 2)
        x1, x2 = max(0, x - self.size // 2), min(w, x + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotune conv algorithms for fixed input size

def main():
    model = ResNet18().to(device, memory_format=torch.channels_last)
    model = torch.compile(model) # backend="aot_eager" (for Windows)

    # Data Augmentation Pipeline
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
        Cutout(size=16),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=1,
        pin_memory=True, persistent_workers=True, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=1, pin_memory=True
    )

    # Split params: no weight decay on BN params or biases
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or 'bn' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.SGD([
        {'params': decay, 'weight_decay': 5e-4},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=args.lr, momentum=0.9, nesterov=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(trainloader), pct_start=0.3
    )

    torch.cuda.synchronize()
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        print(f'Epoch {epoch+1}/{args.epochs} | Loss: {running_loss/total:.4f} | '
            f'Train Acc: {100.*correct/total:.2f}% | LR: {optimizer.param_groups[0]['lr']}')

    torch.cuda.synchronize()
    train_time = time.time() - start

    # Evaluate model test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
    test_acc = 100. * correct / total

    print(f'Training time: {train_time:.1f}s | Test Acc: {test_acc:.2f}%')

    save_checkpoint(
        model, optimizer, epoch+1, running_loss/total,
        args.data_dir / f"resnet_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt",
    )


if __name__ == "__main__":
    main()