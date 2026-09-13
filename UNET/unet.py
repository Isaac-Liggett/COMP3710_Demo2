import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d
from torch.nn.functional import relu, softmax, one_hot
from torchvision import models

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # Added padding to keep dimensions clean
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder block (Using ConvTranspose2d to upsample spatial dimensions)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d12 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # 512 (up) + 512 (shortcut) = 1024 channels
        self.d13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d22 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 256 + 256 = 512 channels
        self.d23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d32 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 128 + 128 = 256 channels
        self.d33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d42 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 + 64 = 128 channels
        self.d43 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output Convolutional layer
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder with correct upsampling and concatenation shortcuts
        xd11 = self.up1(xe52)
        xu11 = torch.cat([xd11, xe42], dim=1)
        xd12 = relu(self.d12(xu11))
        xd13 = relu(self.d13(xd12))

        xd21 = self.up2(xd13)
        xu22 = torch.cat([xd21, xe32], dim=1)
        xd22 = relu(self.d22(xu22))
        xd23 = relu(self.d23(xd22))

        xd31 = self.up3(xd23)
        xu32 = torch.cat([xd31, xe22], dim=1)
        xd32 = relu(self.d32(xu32))
        xd33 = relu(self.d33(xd32))

        xd41 = self.up4(xd33)
        xu42 = torch.cat([xd41, xe12], dim=1)
        xd42 = relu(self.d42(xu42))
        xd43 = relu(self.d43(xd42))

        return self.outconv(xd43)
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits:  Raw model outputs, shape (B, C, H, W)
            targets: One-hot float labels, shape (B, C, H, W)
        """
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")

        probs = softmax(logits, dim=1)

        # Sum over spatial dims only, keep batch and class separate.
        # dims = (0, 2, 3) reduces over batch AND space, giving one dice score per class
        # averaged across the batch. Reducing over batch here (not treating each
        # sample independently) is a modeling choice — see note below.
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()
