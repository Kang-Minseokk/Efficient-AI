class TernaryLinear(nn.Module):
    """
    BitNet TernaryLinear: 2-bit (1.58-bit) weight quantization with per-tensor scale α.
    Forward: w_q = STE( clamp(round(w/α), -1,1) * α ), y = x @ w_q^T + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.norm = nn.RMSNorm(in_features, eps=1e-8)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = self.norm(x)
        w_q = self.weight + (ter_weight_quant(self.weight) - self.weight).detach()
        return F.linear(x, w_q, self.bias)