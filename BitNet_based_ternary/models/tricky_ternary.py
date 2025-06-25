from models.bit1 import BinaryLinear

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import RMSNorm


class TernaryTrickLinear(nn.Module):
    def __init__(self,
                 in_features: int = 28 * 28,
                 hidden_features: int = 256,
                 bias=True,
                 ):
        super().__init__()
        self.fc1 = BinaryLinear(in_features, hidden_features, bias=bias)
        self.fc2 = BinaryLinear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) :        
        return (self.fc1(x) - self.fc2(x))*0.5


class TerTrickMLP(nn.Module):
    """
    BitNet-style MLP for MNIST with adjustable depth (default 4).
    Each hidden layer is a BinaryLinear or TernaryLinear quantized layer.
    """    
    def __init__(self,
                 in_features: int = 32 * 32 * 3, 
                 hidden_features: int = 256,
                 num_classes: int = 10,
                 depth: int = 4,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        LinearLayer = TernaryTrickLinear
        layers = []
        # Input layer
        layers.append(nn.Flatten())
        # Hidden layers
        for i in range(depth):
            in_f = in_features if i == 0 else hidden_features
            layers.append(LinearLayer(in_f, hidden_features, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(LinearLayer(hidden_features, num_classes, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)