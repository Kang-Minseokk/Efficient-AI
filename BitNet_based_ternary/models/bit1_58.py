import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import RMSNorm

class TernaryLinear(nn.Module):
    """
    BitNet TernaryLinear: 2-bit (1.58-bit) weight quantization with per-tensor scale α.
    Forward: w_q = STE( clamp(round(w/α), -1,1) * α ), y = x @ w_q^T + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.norm = RMSNorm(in_features, eps=1e-8)        

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def ter_weight_quant(self, weight):
        alpha = torch.mean(torch.abs(weight)) # 위에서 Learnable Parameter로 설정해보았습니다.
        result = torch.clamp(torch.round(weight / alpha), -1, 1) * alpha
        return result

    def forward(self, x):
        x = self.norm(x)
        w_q = self.weight + (self.ter_weight_quant(self.weight) - self.weight).detach()
        
        return F.linear(x, w_q, self.bias)

# Scailng Factor를 Learnable Parameter로 설정하는 Ternary Linear Class입니다.
    
# class TernaryLinear(nn.Module):
#     """
#     BitNet TernaryLinear: 2-bit (1.58-bit) weight quantization with per-tensor scale α.
#     Forward: w_q = STE( clamp(round(w/α), -1,1) * α ), y = x @ w_q^T + b
#     """
#     def __init__(self, in_features, out_features, bias=True):
#         super().__init__()
#         self.in_features  = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.alpha = nn.Parameter(torch.ones(1))
#         self.reset_parameters()
#         self.norm = RMSNorm(in_features, eps=1e-8)
        

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             bound = 1.0 / math.sqrt(self.in_features)
#             nn.init.uniform_(self.bias, -bound, bound)
        
#         with torch.no_grad():
#             alpha_val = 0.7 * torch.mean(torch.abs(self.weight))
#             self.alpha.copy_(alpha_val)
            
#     def ter_weight_quant(self, weight):
#         alpha = torch.mean(torch.abs(weight)) # 위에서 Learnable Parameter로 설정해보았습니다.
#         result = torch.clamp(torch.round(weight / self.alpha), -1, 1) * self.alpha
#         return result

#     def forward(self, x):
#         x = self.norm(x)
#         w_q = self.weight + (self.ter_weight_quant(self.weight) - self.weight).detach()
        
#         return F.linear(x, w_q, self.bias)
    
    
class TernaryMLP(nn.Module):
    """
    BitNet-style MLP for MNIST with adjustable depth (default 4).
    Each hidden layer is a BinaryLinear or TernaryLinear quantized layer.
    """    
    def __init__(self,
                 in_features: int = 32 * 32 * 3, 
                 hidden_features: int = 512,
                 num_classes: int = 10,
                 depth: int = 4,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        LinearLayer = TernaryLinear
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