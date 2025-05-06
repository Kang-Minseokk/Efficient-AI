import torch
import torch.nn as nn

def sign_function(x):
    return torch.sign(torch.sign(x) + 0.5)

class SignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = sign_function(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (torch.abs(x) <= 1).to(torch.float32)

class SignFunctionNoSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = sign_function(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        return grad

class SignModule(nn.Module):
    def __init__(self):
        super().__init__()        
        self.func = SignFunction.apply

    def forward(self, x):
        return self.func(x)
    
class TrackedSignModule(nn.Module): # Activation을 저장해주는 기능을 추가한 모듈입니다. (minseok) - 전체 수정함
    def __init__(self, activation_store_list=None):
        super().__init__()
        self.activation_store_list = activation_store_list        

    def forward(self, x):
        out = torch.sign(x)                
        if self.activation_store_list is not None:
            self.activation_store_list.append(out.detach().clone())
        return out

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.conv2d = (lambda x, w, b: nn.functional.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups))
        self.weight = nn.parameter.Parameter(data=torch.zeros(out_channels, round(in_channels/groups), kernel_size, kernel_size), requires_grad=True)
        self.bias = nn.parameter.Parameter(data=torch.zeros(out_channels), requires_grad=True) if bias else None

    def forward(self, x):
        b_weight = SignFunctionNoSTE.apply(self.weight)
        b_bias = SignFunctionNoSTE.apply(self.bias) if self.bias is not None else self.bias
        return self.conv2d(x, b_weight, b_bias)

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.linear = nn.functional.linear
        self.weight = nn.parameter.Parameter(data=torch.zeros(out_features, in_features), requires_grad=True)
        self.bias = nn.parameter.Parameter(data=torch.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x):
        b_weight = SignFunctionNoSTE.apply(self.weight)
        b_bias = SignFunctionNoSTE.apply(self.bias) if self.bias is not None else self.bias
        return self.linear(x, b_weight, b_bias)