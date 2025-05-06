import torch
import torch.nn as nn
from models.base import *

class VGG(nn.Module):
    def __init__(self, cfg, bias=False, batch_norm=True, batch_norm_out=False, affine=False, b_act=False, b_weight=False, num_classes=200):
        super().__init__()
        Linear = BinaryLinear if b_weight else nn.Linear
        Activation = SignModule if b_act else nn.ReLU
        BatchNorm = nn.BatchNorm1d if batch_norm else nn.Identity

        self.features = self._make_layers(cfg, bias, batch_norm, affine, b_act, b_weight)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            Linear(512 * 7 * 7, 4096, bias=bias),                       # No. 0
            BatchNorm(4096, affine=affine, track_running_stats=False),  # No. 1
            Activation(),
            Linear(4096, 4096, bias=bias),                              # No. 3
            BatchNorm(4096, affine=affine, track_running_stats=False),  # No. 4
            Activation(),
            Linear(4096, num_classes, bias=bias)                        # No. 6
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, BinaryConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                #nn.init.normal_(m.weight, 0, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, BinaryLinear):
                nn.init.normal_(m.weight, 0, 1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, bias, batch_norm, affine, b_act, b_weight):
        layers = []
        in_channels = cfg[0]  # in_channels

        for i, v in enumerate(cfg[1:]):
            if v=="M":
                continue
            else:
                Activation = SignModule if b_act else nn.ReLU
                BatchNorm = nn.BatchNorm2d if batch_norm else nn.Identity
                Conv2d = BinaryConv2d if b_weight else nn.Conv2d
                c2 = Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)

                layers += [c2]
                if i+2<len(cfg) and cfg[i+2]=="M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers += [BatchNorm(v, affine=affine, track_running_stats=False), Activation()]

                in_channels = v

        return nn.Sequential(*layers)


cfgs = {
    "A": ["M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def get_vgg(mode="A", bias=False, batch_norm=True, batch_norm_out=False, affine=False, b_act=False, b_weight=False, in_shape=(64,3), out_dim=200):
    in_dim = 64  # in_shape[0]
    in_channels = in_shape[1]
    return VGG([in_channels, in_dim]+cfgs[mode], bias=bias, batch_norm=batch_norm, batch_norm_out=batch_norm_out, affine=affine,
                b_act=b_act, b_weight=b_weight, num_classes=out_dim)