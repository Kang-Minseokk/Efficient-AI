from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from get_args import get_args

from models.bit1 import BinaryLinear # Binary Linear Import 
from models.bit1_58 import TernaryLinear
from models.tricky_ternary import TernaryTrickLinear

pair = lambda x : x if isinstance(x, tuple) else (x, x)

# 양자화 타입을 설정하는 인자를 통해서 선형 레이어(LinearLayer)를 결정합니다.
args = get_args()
if args.quantize_type == 'binary' :
    print("mlp mixer - Binary")
    LinearLayer = BinaryLinear
elif args.quantize_type == 'real' :
    print("mlp mixer - Real")
    LinearLayer = nn.Linear
elif args.quantize_type == 'qat_ternary' :
    print("mlp mixer - Qat Ternary")
    LinearLayer = TernaryLinear
elif args.quantize_type == 'trick_ternary' :
    print("mlp mixer - Trick Ternary")
    LinearLayer = TernaryTrickLinear

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = LinearLayer):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )
    
def MLPMixer(*, image_size, channels, patch_size, dim, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., depth):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), LinearLayer
    
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        LinearLayer((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        LinearLayer(dim, num_classes)
    )
    