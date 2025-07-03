import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from models.utils_quant import BitLinear
from models.configuration_bitnet import BitnetConfig

class BitnetRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class BitnetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.weight_bits = config.weight_bits        
        self.gate_proj = BitLinear(
            self.hidden_size, self.intermediate_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.up_proj = BitLinear(
            self.hidden_size, self.intermediate_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.down_proj = BitLinear(
            self.intermediate_size, self.hidden_size, bias=False, 
            weight_bits=config.weight_bits, input_bits=config.input_bits, 
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.ffn_layernorm = BitnetRMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
    # 아래 한 줄의 코드가 없으면, Pytorch의 Broadcast 기능 때문에 차원 mismatch 오류가 발생한다.
        x = x.view(x.size(0), -1) 
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_layernorm(x)
        x = self.down_proj(x)
        return x