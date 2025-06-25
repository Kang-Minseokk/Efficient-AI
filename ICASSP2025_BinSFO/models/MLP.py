import os
import pdb
import pickle
import torch
import torch.nn as nn
from models.base import *


class MLP(nn.Module):
    def __init__(self, in_dim=12288, hid_dim=8192, out_dim=200, depth=4, approx=None, use_ternary=False, qat_ternary=False):
        super(MLP, self).__init__()
        self.depth = depth
        
        def get_linear_class(use_ternary, qat_ternary):
            if qat_ternary:
                print("Layer Type: qat ternary\n")
                return QatTernaryLinear
            elif use_ternary:
                print("Layer Type: Tricky ternary\n")
                return TernaryLinear
            else:
                print("Layer Type: Binary\n")
                return BinaryLinear

        layers = []
        # MLP의 Layer 결정
        LinearClass = get_linear_class(use_ternary, qat_ternary)
        for d in range(depth-1):
            dim1 = in_dim if d==0 else hid_dim
            dim2 = hid_dim                        
            # 기존 Layer:
            layers += [LinearClass(dim1, dim2),
                       nn.SyncBatchNorm(dim2, affine=False, track_running_stats=False),
                       BinaryActivation(approx=approx)]
            # layers += [LinearClass(dim1, dim2),
            #            nn.GELU(),
            #            nn.Dropout(0.1) # Dropout 레이어 추가
            #            ]
        self.feature = nn.Sequential(*layers)

        dim1 = hid_dim if depth>1 else in_dim
        out_layer = [LinearClass(dim1, out_dim),
                     nn.SyncBatchNorm(out_dim, affine=False, track_running_stats=False)]
        self.classifier = nn.Sequential(*out_layer)

        self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.feature(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BinaryLinear) or (isinstance(m, nn.Linear)):
                nn.init.normal_(m.weight, 0, 1e-2)

            elif isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_rectified_scale(self, t):
        assert 0<=t<=1
        for i in range(self.depth-1):
            self.feature[3*i+2].scale = torch.tensor(1+t*2).float()


def get_mlp(in_shape=(32,3), hid_dim=512, out_dim=10, depth=4, approx=None, use_ternary=False, qat_ternary=False):
    in_dim = in_shape[0]**2 * in_shape[1] # Slightly change for CIFAR10 in lower resolution
    
    model = MLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, depth=depth, approx=approx, use_ternary=use_ternary, qat_ternary = qat_ternary)
    
    return model
