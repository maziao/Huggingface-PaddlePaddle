import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class FastGELUActivation(Activation):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return 0.5 * hidden_states * (1.0 + paddle.nn.functional.tanh(
            hidden_states * 0.7978845608 * (1.0 + 0.044715 * hidden_states * hidden_states)))
