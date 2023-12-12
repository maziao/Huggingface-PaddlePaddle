import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@dataclass
class ClippedGELUConfig(BaseConfig):
    min_val: int = -10
    max_val: int = 10


@ACTIVATION.register_module
class ClippedGELUActivation(Activation):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """
    config_class = ClippedGELUConfig

    def __init__(self, config: ClippedGELUConfig):
        super().__init__(config)
        if config.min_val > config.max_val:
            raise ValueError(
                f'min should be < max (got min: {config.min_val}, max: {config.max_val})'
            )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.clip(paddle.nn.functional.gelu(x=hidden_states), min=self.config.min_val, max=self.config.max_val)
