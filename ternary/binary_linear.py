import torch

class BinaryLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        binary_weight = weight.sign() # 가중치의 부호만 사용해봅니다.
        output = input @ binary_weight.t()
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output @ weight.sign()
        grad_weight = grad_output.t() @ input
        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_input, grad_weight, grad_bias
