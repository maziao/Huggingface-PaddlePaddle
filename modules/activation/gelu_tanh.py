import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class GELUTanhActivation(Activation):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """
    config_class = PseudoConfig

    def __init__(self, config: PseudoConfig):
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.gelu(x=hidden_states, approximate=True)
