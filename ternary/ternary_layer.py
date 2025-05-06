import torch
import torch.nn as nn
from binary_layer import BinaryLinear

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
        output1 = self.binary1(input)
        output2 = self.binary2(input)
        output = output1 - output2
        if self.bias is not None:
            output += self.bias
        return output
