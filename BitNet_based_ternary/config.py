from transformers import PretrainedConfig

class BitnetConfig(PretrainedConfig):
    def __init__(
        self, 
        hidden_size = 32,
        intermediate_size = 24576,
        hidden_act = "gelu", 
        rms_norm_eps = 1e-6,
        num_classes = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.num_classes = num_classes