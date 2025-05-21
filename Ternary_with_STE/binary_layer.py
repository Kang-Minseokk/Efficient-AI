import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from binary_linear import BinaryLinearFunction

class BinaryLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return BinaryLinearFunction.apply(input, self.weight, self.bias)
    
    def get_binary_weight(self):
        return self.weight.sign()
