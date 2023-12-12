import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class MishActivation(Activation):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.mish(hidden_states)
