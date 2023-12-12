import math
import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class NewGELUActivation(Activation):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return 0.5 * hidden_states * (1.0 + paddle.nn.functional.tanh(
            math.sqrt(2.0 / math.pi) * (hidden_states + 0.044715 * paddle.pow(hidden_states, 3.0))))
