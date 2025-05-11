import torch
import torch.nn as nn
from binary_layer import BinaryLinear
from ternary_linear import TernaryLinearFunction

class TernaryLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.binary1 = BinaryLinear(input_features, output_features, bias=False)
        self.binary2 = BinaryLinear(input_features, output_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return TernaryLinearFunction.apply(
            input,
            self.binary1.weight,
            self.binary2.weight,
            self.bias
        )

