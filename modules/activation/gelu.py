import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class GELUActivation(Activation):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in `nn.functional`
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    config_class = PseudoConfig

    def __init__(self, config: PseudoConfig):
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.gelu(x=hidden_states, approximate=False)
