import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import RMSNorm
from torch.nn.modules.utils import _pair
    
# Scaled Binary Linear Layer
class ScaleBinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, positive_ratio=0.5):
        super().__init__()
        self.positive_ratio = positive_ratio
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.scale = nn.Parameter(torch.full((1,), math.log(1.0/math.sqrt(in_features))))
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        w_q = self.weight + self.weight.sign().detach() * torch.exp(self.scale) - self.weight.detach()
        return F.linear(x, w_q)
    


class TernaryTrickLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias=True,
                 use_norm=False,
                 zero_ratio=0.25
                 ):
        super().__init__()
        self.zero_ratio = zero_ratio
        self.fc1 = ScaleBinaryLinear(in_features, hidden_features)
        self.fc2 = ScaleBinaryLinear(in_features, hidden_features)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_features))
        else :
            self.register_parameter('bias', None)              
        
        if use_norm :
            self.norm = RMSNorm(in_features, eps=1e-8)
        else :
            self.norm = None
            
        # self.set_positive_ratio(w1 = self.fc1.weight, w2 = self.fc2.weight)
        
    def set_positive_ratio(self, w1, w2):
        """두 가중치가 서로 반대가 되도록 편향시켜 초기화"""
        with torch.no_grad():
            total_params = self.fc1.weight.numel()
            
            # 원하는 0의 개수
            num_zeros = int(total_params * self.zero_ratio)
            
            # 랜덤하게 0이 될 위치 선택
            zero_indices = torch.randperm(total_params)[:num_zeros]
            
            # 모든 위치를 flatten하여 처리
            w1_flat = w1.view(-1)
            w2_flat = w2.view(-1)
            
            # 0이 될 위치는 같은 값으로 설정 (sign이 같아지도록)
            for idx in zero_indices:
                sign_val = torch.randint(0, 2, (1,)).item() * 2 - 1  # -1 or 1
                w1_flat[idx] = sign_val * abs(w1_flat[idx])
                w2_flat[idx] = sign_val * abs(w2_flat[idx])
            
            # 나머지 위치는 반대 부호가 되도록 편향
            non_zero_mask = torch.ones(total_params, dtype=torch.bool)
            non_zero_mask[zero_indices] = False
            
            for idx in torch.where(non_zero_mask)[0]:
                # 반대 부호가 되도록 편향 (80% 확률로 반대)                
                if torch.rand(1).item() < 0.5:
                    w1_flat[idx] = abs(w1_flat[idx])
                    w2_flat[idx] = -abs(w2_flat[idx])
                else:
                    w1_flat[idx] = -abs(w1_flat[idx])
                    w2_flat[idx] = abs(w2_flat[idx])
            
            # 원래 shape로 복원
            self.fc1.weight.data = w1_flat.reshape(self.fc1.weight.shape)
            self.fc2.weight.data = w2_flat.reshape(self.fc2.weight.shape)
    

    def forward(self, x: torch.Tensor, layer_name: str = "Layer", step: int = None) :
        if self.norm is not None:
            x = self.norm(x)
        
        out1 = self.fc1(x)
        out2 = self.fc2(x)
                
        out = out1 - out2
        
        # BitNet 스타일 scaling factor
        with torch.no_grad():
            ternary_real_weight = self.fc1.weight - self.fc2.weight
            scale = ternary_real_weight.abs().mean()
            scale = scale.clamp(min=1e-5)

        out = out * self.alpha
        
        if self.bias is not None:
            out = out + self.bias
            
        
        # Trick Ternary 가중치 비율 확인용    
        w1_bin = self.fc1.weight.data.sign()
        w2_bin = self.fc2.weight.data.sign()
        ternary_weight = (w1_bin - w2_bin) * self.alpha.data
        
        num_pos1 = (ternary_weight > 0).sum().item()
        num_zero = (ternary_weight == 0).sum().item()
        num_neg1 = (ternary_weight < 0).sum().item()
        total = ternary_weight.numel()
        
        # 구분 정보 출력
        header = f"[{layer_name}]"
        if step is not None:
            header += f" Step: {step}"
        
        # print(f"Proportion of +1: {num_pos1 / total:.2%}")
        # print(f"Proportion of  0: {num_zero / total:.2%}")
        # print(f"Proportion of -1: {num_neg1 / total:.2%}")
        
        return out
    

class BinaryConv1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.scale = nn.Parameter(torch.full((1,), math.log(1.0/math.sqrt(in_features))))
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        w_q = self.weight + self.weight.sign().detach() * torch.exp(self.scale) - self.weight.detach()
        # print("x dim:", x.size())
        
        # print("bias dim:", self.bias.size() if self.bias is not None else "No bias")
        return F.conv1d(x, w_q, self.bias, self.stride, self.padding)
    
    
class TernaryTrickConv1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding                        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.alpha = nn.Parameter(torch.tensor(0.5))    
        self.reset_parameters()
        
        self.conv1 = BinaryConv1d(in_features, out_features, kernel_size, stride, padding, bias)
        self.conv2 = BinaryConv1d(in_features, out_features, kernel_size, stride, padding, bias)
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = (out1 - out2) * self.alpha        
        return out + self.bias if self.bias is not None else out
    
    
class BinaryConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, kernel_size, kernel_size))
        if bias:                        
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.scale = nn.Parameter(torch.full((1,), math.log(1.0/math.sqrt(in_features * self.kernel_size * self.kernel_size))))
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features * self.kernel_size * self.kernel_size)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        w_q = self.weight + self.weight.sign().detach() * torch.exp(self.scale) - self.weight.detach()       
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)

    
class TernaryTrickConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding=0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.conv1 = BinaryConv2d(in_features, out_features, kernel_size, stride, padding, bias)
        self.conv2 = BinaryConv2d(in_features, out_features, kernel_size, stride, padding, bias)
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features, 1 , 1))  # 하드 코딩을 해버림..
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))        
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.out_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = (out1 - out2) * self.alpha        
        return out + self.bias if self.bias is not None else out