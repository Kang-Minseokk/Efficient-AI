# utils.py
import torch
import torch.nn as nn

# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-8):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / x.shape[-1]**0.5)
#         return self.weight * (x / (norm + self.eps))

# BitNet 논문 기반 RMSNorm 클래스 구현
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BitnetRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)