# test.py
import torch
from models.MLP import BinaryLinear  # 이미 구현되어 있는 이진 레이어
from torch import nn

# TernaryLinear 구현 (이전에 공유한 코드)
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, xnor=False):
        super(TernaryLinear, self).__init__()
        self.binary1 = BinaryLinear(in_features, out_features, xnor=xnor)
        self.binary2 = BinaryLinear(in_features, out_features, xnor=xnor)
        # 테스트를 위한 임시 강제 초기화
        with torch.no_grad():
            self.binary1.weight.fill_(1.0)    # sign -> +1
            self.binary2.weight.fill_(-1.0)   # sign -> -1

    def forward(self, x):
        out1 = self.binary1(x)
        out2 = self.binary2(x)
        return out1 - out2

# 테스트 코드
if __name__ == "__main__":
    model = TernaryLinear(in_features=4, out_features=2)
    x = torch.randn(1, 4)
    y = model(x)
    print("Output:", y)
    print("Unique (rounded) values:", torch.unique(y.round()))
