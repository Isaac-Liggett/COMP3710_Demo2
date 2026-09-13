from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class OasisSegmentationDataset(Dataset):
    def __init__(self, data_dir, label_dir, image_size=64):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)

        self.data_filenames = sorted([p.name for p in self.data_dir.rglob("*") if p.suffix.lower() == ".png"])
        self.label_filenames = sorted([p.name for p in self.label_dir.rglob("*") if p.suffix.lower() == ".png"])

        if len(self.data_filenames) != len(self.label_filenames):
            raise RuntimeError(f"The directories have different numbers of files!")

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        self.unique_colours = [0, 85, 170, 255]

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        img_path = self.data_dir / self.data_filenames[idx]
        label_path = self.label_dir / self.label_filenames[idx]

        # Load images
        img = Image.open(img_path).convert("L")
        label = Image.open(label_path).convert("L")

        # Apply transforms
        img = self.img_transform(img)
        label = self.label_transform(label)

        label = np.array(label)

        class_mask = np.zeros_like(label, dtype=np.int64)
        for class_idx, colour_val in enumerate(self.unique_colours):
            class_mask[label == colour_val] = class_idx
        
        class_tensor = torch.from_numpy(class_mask).long()
        
        one_hot = torch.nn.functional.one_hot(class_tensor, num_classes=4).permute(2, 0, 1)

        return img, one_hot

def get_dataloader(data_dir, label_dir, image_size=64, batch_size=64, num_workers=4, shuffle=True):
    dataset = OasisSegmentationDataset(data_dir, label_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
