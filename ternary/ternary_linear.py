import torch

class TernaryLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w1, w2, bias=None):
        ctx.save_for_backward(input, w1, w2, bias)
        ternary_weight = torch.sign(w1 - w2)
        output = input @ ternary_weight.T
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w1, w2, bias = ctx.saved_tensors

        # STE: pretend ternary_weight = w1 - w2
        grad_input = grad_output @ torch.sign(w1 - w2)  

        # 여기서 핵심: w1에 대해서는 +, w2에 대해서는 -
        grad_w1 = grad_output.T @ input  # ∂L/∂(w1)
        grad_w2 = -grad_output.T @ input  # ∂L/∂(w2) (음의 방향으로 업데이트)

        grad_bias = grad_output.sum(dim=0) if bias is not None else None
        return grad_input, grad_w1, grad_w2, grad_bias
