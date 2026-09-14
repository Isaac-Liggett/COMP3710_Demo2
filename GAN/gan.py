import torch
import torch.nn as nn

class Generator(nn.Module): 
    def __init__(self, image_size=64):
        super().__init__()

        self.image_size = image_size
        
        self.linear1 = nn.Linear(1, 256)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(256, 784)
        
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.tanh(x)
        
        x = x.view(-1, self.image_size, self.image_size)
        
        return x

class Discriminator(nn.Module): 
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.linear2 = nn.Linear(256, 1)
        self.dropout2 = nn.Dropout(0.5)
        
        # Use sigmoid to ensure output is a probability for the loss function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.linear2(x)
        
        x = self.sigmoid(x)
        
        return x