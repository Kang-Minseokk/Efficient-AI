import torch
import torch.nn as nn
from models.base import *

class MLP(nn.Module):
    def __init__(self, bias=False, batch_norm=True, batch_norm_out=False, affine=False, b_act=False, b_weight=False, training_space="binary",
                 in_dim=12288, hid_dim=8192, out_dim=200, depth=4):
        super().__init__()
        self.activations = [] # minseok        
        Linear = BinaryLinear if b_weight else nn.Linear
        Activation = (lambda: TrackedSignModule(activation_store_list=self.activations)) if b_act else nn.ReLU # minseok
        BatchNorm = nn.BatchNorm1d if batch_norm else nn.Identity

        layers = []
        for d in range(depth-1):
            dim1 = in_dim if d==0 else hid_dim
            dim2 = hid_dim
            layers += [Linear(dim1, dim2, bias=bias),
                       BatchNorm(dim2, affine=affine, track_running_stats=False),
                       Activation()]
        if batch_norm_out:
            layers += [Linear(hid_dim, out_dim, bias=bias), BatchNorm(out_dim, affine=False, track_running_stats=False)]
        else:
            layers += [Linear(hid_dim, out_dim, bias=bias)]
        
        self.classifier = nn.Sequential(*layers)
        self._initialize_weights(training_space)
        

    def forward(self, x):
        self.activations.clear() # 매번 초기화해서 이전 값을 제거합니다. 
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, training_space):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, BinaryLinear):
                nn.init.normal_(m.weight, 0, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
                if training_space=="binary":
                    m.weight.data = sign_function(m.weight.data)
                    if m.bias is not None:
                        m.bias.data = sign_function(m.bias.data)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def get_mlp(bias=False, batch_norm=True, batch_norm_out=False, affine=False, b_act=False, b_weight=False, training_space="real",
            in_shape=(64,3), hid_dim=256, out_dim=200, depth=4):
    in_dim = in_shape[0]**2 * in_shape[1]
    return MLP(bias=bias, batch_norm=batch_norm, batch_norm_out=batch_norm_out, affine=affine, b_act=b_act, b_weight=b_weight, training_space=training_space,
               in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, depth=depth)