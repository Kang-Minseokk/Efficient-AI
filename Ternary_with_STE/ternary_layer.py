import torch
import torch.nn as nn
from binary_layer import BinaryLinear

class TernaryLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super().__init__()
        self.binary1 = BinaryLinear(input_features, output_features, bias=False)
        self.binary2 = BinaryLinear(input_features, output_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        output = self.binary1(input) - self.binary2(input)
        if self.bias is not None:
            output += self.bias
        return output

