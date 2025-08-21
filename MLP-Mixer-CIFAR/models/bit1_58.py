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
        alpha = torch.clamp(torch.mean(torch.abs(weight)), min=1e-5)
        result = torch.clamp(torch.round(weight / alpha), -1, 1) * alpha
        return result

    def forward(self, x):
        w_q = self.weight + (self.ter_weight_quant(self.weight) - self.weight).detach()
        
        # num_pos1 = (w_q > 0).sum().item()
        # num_zero = (w_q == 0).sum().item()
        # num_neg1 = (w_q < 0).sum().item()
        # total = w_q.numel()
        
        # print(f"Proportion of +1: {num_pos1 / total:.2%}")
        # print(f"Proportion of  0: {num_zero / total:.2%}")
        # print(f"Proportion of -1: {num_neg1 / total:.2%}")
        
        return F.linear(x, w_q, self.bias)
    
    
class TernaryConv1d(nn.Module):
    def __init__(self, in_features, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_features, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def ter_weight_quant(self, weight):
        alpha = torch.clamp(torch.mean(torch.abs(weight)), min=1e-5)
        result = torch.clamp(torch.round(weight / alpha), -1, 1) * alpha
        return result
    
    def forward(self, x):
        w_q = self.weight + (self.ter_weight_quant(self.weight) - self.weight).detach()
        return F.conv1d(x, w_q, self.bias, self.stride, self.padding)
    
    
class TernaryConv2d(nn.Module):
    def __init__(self, in_features, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_features, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def ter_weight_quant(self, weight):
        alpha = torch.clamp(torch.mean(torch.abs(weight)), min=1e-5)
        result = torch.clamp(torch.round(weight / alpha), -1, 1) * alpha
        return result
    
    def forward(self, x):
        w_q = self.weight + (self.ter_weight_quant(self.weight) - self.weight).detach()
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)