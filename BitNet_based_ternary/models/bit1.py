import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import RMSNorm


class BinaryLinear(nn.Module):
    """
    BitNet BinaryLinear: 1-bit weight quantization with per-tensor scale α.
    Forward: w_q = STE( sign(w) * α ), y = x @ w_q^T + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        # latent weight: high-precision parameter
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.norm = RMSNorm(in_features, eps=1e-8)
        self.scale = nn.Parameter(torch.full((1,), math.log(1.0/math.sqrt(in_features))))
        print(self.weight.abs().mean(), torch.exp(self.scale))

    def reset_parameters(self):
        # 동일하게 nn.Linear 초기화
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.norm(x)
        w_q = self.weight + self.weight.sign().detach() * torch.exp(self.scale) - self.weight.detach()
        # -1과 1을 가지는 이진 가중치에 scaling factor를 곱하여 크기를 반영할 수 있도록 합니다.        
        return F.linear(x, w_q, self.bias)

class BitNetMLP(nn.Module):
    """
    BitNet-style MLP for MNIST with adjustable depth (default 4).
    Each hidden layer is a BinaryLinear or TernaryLinear quantized layer.
    """
    def __init__(self,
                 in_features: int = 32*32*3,
                 hidden_features: int = 512,
                 num_classes: int = 10,
                 depth: int = 4,
                 dropout: float = 0.1,                 
                 ):
        super().__init__()

        layers = []
        # Input layer
        layers.append(nn.Flatten())
        # Hidden layers
        for i in range(depth):
            in_f = in_features if i == 0 else hidden_features
            layers.append(BinaryLinear(in_f, hidden_features, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(BinaryLinear(hidden_features, num_classes, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    