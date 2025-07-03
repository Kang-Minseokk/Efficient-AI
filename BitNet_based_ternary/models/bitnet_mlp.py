from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from get_args import get_args

from models.bit1 import BinaryLinear # Binary Linear Import 
from models.bit1_58 import TernaryLinear
from models.tricky_ternary import TernaryTrickLinear

# For BitNet Style
from utils import RMSNorm
from transformers.activations import ACT2FN

pair = lambda x : x if isinstance(x, tuple) else (x, x)

# 양자화 타입을 설정하는 인자를 통해서 선형 레이어(LinearLayer)를 결정합니다.
args = get_args()
if args.quantize_type == 'binary' :
    print("bitnet_mlp - Binary")
    LinearLayer = BinaryLinear
elif args.quantize_type == 'real' :
    print("bitnet_mlp - Real")
    LinearLayer = nn.Linear
elif args.quantize_type == 'qat_ternary' :
    print("bitnet_mlp - Qat Ternary")
    LinearLayer = TernaryLinear
elif args.quantize_type == 'trick_ternary' :
    print("bitnet_mlp - Trick Ternary")
    LinearLayer = TernaryTrickLinear
    
class BitnetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = LinearLayer(
            self.hidden_size, self.intermediate_size, bias = False, 
        )
        
        self.up_proj = LinearLayer(
            self.hidden_size, self.intermediate_size, bias = False,
        )
        
        self.down_proj = LinearLayer(
            self.intermediate_size, self.hidden_size, bias = False,
        )
        
        self.act_fn = ACT2FN[config.hidden_act]
        self.ffn_layernorm = RMSNorm(self.intermediate_size, eps = config.rms_norm_eps)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(3, config.num_classes)
        )
        
    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_layernorm(x)
        x = self.down_proj(x)
        x = self.classifier(x)        
        return x