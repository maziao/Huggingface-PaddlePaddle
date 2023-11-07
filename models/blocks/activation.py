import paddle
import math
from packaging import version
from abc import abstractmethod
from models.registry import ACTIVATION
from dataclasses import dataclass
from config.base import BaseConfig


@dataclass
class ActivationConfig(BaseConfig):
    use_gelu_python: bool = True
    max_val: int = 10
    min_val: int = -10
    in_place: bool = False


class Activation(paddle.nn.Layer):
    config_class = ActivationConfig

    def __init__(self, config: ActivationConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError


@ACTIVATION.register_module
class PaddleGELUTanh(Activation):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, config: ActivationConfig):
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.gelu(x=hidden_states, approximate=True)


@ACTIVATION.register_module
class NewGELUActivation(Activation):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return 0.5 * hidden_states * (1.0 + paddle.nn.functional.tanh(
            math.sqrt(2.0 / math.pi) * (hidden_states + 0.044715 * paddle.pow(hidden_states, 3.0))))


@ACTIVATION.register_module
class GELUActivation(Activation):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in `nn.functional`
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, config: ActivationConfig):
        super().__init__(config)
        if self.config.use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = paddle.nn.functional.gelu

    @staticmethod
    def _gelu_python(hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states * 0.5 * (1.0 + paddle.erf(hidden_states / math.sqrt(2.0)))

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.act(hidden_states)


@ACTIVATION.register_module
class FastGELUActivation(Activation):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return 0.5 * hidden_states * (1.0 + paddle.nn.functional.tanh(
            hidden_states * 0.7978845608 * (1.0 + 0.044715 * hidden_states * hidden_states)))


@ACTIVATION.register_module
class QuickGELUActivation(Activation):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states * paddle.nn.functional.sigmoid(1.702 * hidden_states)


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

    def __init__(self, config: ActivationConfig):
        super().__init__(config)
        if self.config.min_val > self.config.max_val:
            raise ValueError(
                f'min should be < max (got min: {self.config.min_val}, max: {self.config.max_val})'
            )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.clip(paddle.nn.functional.gelu(x=hidden_states), min=self.config.min_val, max=self.config.max_val)


@ACTIVATION.register_module
class AccurateGELUActivation(Activation):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self, config: ActivationConfig):
        super().__init__(config)
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return 0.5 * hidden_states * (1 + paddle.nn.functional.tanh(
            self.precomputed_constant * (hidden_states + 0.044715 * paddle.pow(hidden_states, 3))))


@ACTIVATION.register_module
class SiLUActivation(Activation):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.silu(hidden_states)


@ACTIVATION.register_module
class MishActivation(Activation):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self, config: ActivationConfig):
        super().__init__(config)
        if version.parse(paddle.__version__) < version.parse('1.9.0'):
            self.act = self._mish_python
        else:
            self.act = paddle.nn.functional.mish

    @staticmethod
    def _mish_python(hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states * paddle.nn.functional.tanh(paddle.nn.functional.softplus(hidden_states))

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return self.act(hidden_states)


@ACTIVATION.register_module
class LinearActivation(Activation):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states


@ACTIVATION.register_module
class LaplaceActivation(Activation):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, hidden_states: paddle.Tensor, mu: float = 0.707107, sigma: float = 0.282095) -> paddle.Tensor:
        hidden_states = (hidden_states - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + paddle.erf(hidden_states))


@ACTIVATION.register_module
class ReLUSquaredActivation(Activation):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        relu_applied = paddle.nn.functional.relu(hidden_states)
        squared = paddle.square(relu_applied)
        return squared


@ACTIVATION.register_module
class ReLU(Activation):

    def __init__(self, config: ActivationConfig):
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.relu(hidden_states)


@ACTIVATION.register_module
class ReLU6(Activation):

    def __init__(self, config: ActivationConfig) -> None:
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.hardtanh(hidden_states, min=0.0, max=6.0)


@ACTIVATION.register_module
class Sigmoid(Activation):

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.sigmoid(hidden_states)


@ACTIVATION.register_module
class Tanh(Activation):

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.tanh(hidden_states)
