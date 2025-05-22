import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_memory_efficient import SignFunction, SignFunctionReSTE, SignFunctionNoSTE, SignFunctionXNOR


class BinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride, padding, xnor=False, qat_ternary=False):
        super(BinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.01, requires_grad=True)
        self.xnor = xnor
        self.qat_ternary = qat_ternary

    def forward(self, x):
        if self.xnor:
            scaling_factor = torch.mean(torch.abs(self.weight)).detach()
            binary_weights = SignFunctionXNOR.apply(self.weight, scaling_factor)
        elif self.qat_ternary: # QAT 방식 추가
            threshold = 0.05  # 또는 torch.mean(torch.abs(self.weight)).item()
            binary_weight = TernaryQuantizeFunction.apply(self.weight, threshold)
        else:
            binary_weights = SignFunction.apply(self.weight)
        return F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, xnor=False, qat_ternary=False):
        super(BinaryLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand((out_features, in_features)) * 0.01, requires_grad=True)
        self.xnor = xnor
        self.qat_ternary = qat_ternary

    def forward(self, x):
        if self.xnor:
            scaling_factor = torch.mean(torch.abs(self.weight)).detach()
            b_weight = SignFunctionXNOR.apply(self.weight, scaling_factor)
        elif self.qat_ternary: # QAT 방식 추가
            threshold = 0.05  # 또는 torch.mean(torch.abs(self.weight)).item()
            binary_weight = TernaryQuantizeFunction.apply(self.weight, threshold)
        else:            
            b_weight = SignFunction.apply(self.weight)            
        
        return F.linear(x, b_weight, None)


class BinaryActivation(nn.Module):
    def __init__(self, approx):
        super(BinaryActivation, self).__init__()
        self.approx = approx
        self.scale = torch.tensor(1).float()

    def forward(self, x):
        if self.approx == "STE":
            out = SignFunction.apply(x)
        elif self.approx == "ApproxSign":
            out = ApproxSign.apply(x)
        elif self.approx == "ReSTE":
            out = SignFunctionReSTE.apply(x, self.scale)
        else:
            raise NotImplementedError()
        return out


class ApproxSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        result = sign_function(x)
        return result

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        mask1 = (-1 <= x) & (x < 0)
        mask2 = (0 <= x) & (x <= 1)
        tmp = torch.zeros_like(grad)
        tmp[mask1] = 2 + 2*x[mask1]
        tmp[mask2] = 2 - 2*x[mask2]
        x_grad = grad * tmp
        return x_grad


# TernaryLinear 클래스 새로 정의
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.binary1 = BinaryLinear(in_features, out_features)
        self.binary2 = BinaryLinear(in_features, out_features)        

    def forward(self, x):
        out1 = self.binary1(x)
        out2 = self.binary2(x)
        return out1 - out2  # 결과적으로 {-1, 0, 1} 도메인 생성

# TernaryConv 클래스 새롭게 정의
class TernaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride, padding, xnor=False):
        super().__init__()
        self.binaryconv1 = BinaryConv(in_chn, out_chn, kernel_size, stride, padding, xnor=False)
        self.binaryconv2 = BinaryConv(in_chn, out_chn, kernel_size, stride, padding, xnor=False)
        
    def forward(self, x):
        out1 = self.binaryconv1(x)
        out2 = self.binaryconv2(x)
        return out1 - out2  
    
    
# Ternary QAT를 위한 클래스 생성
class TernaryQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        out = input.clone()
        out[input > threshold] = 1
        out[input < -threshold] = -1
        out[(input <= threshold) & (input >= -threshold)] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # STE 방식: gradient를 그대로 흘림
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None  # threshold는 학습 대상이 아니므로 None
