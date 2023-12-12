import paddle
from modules.norm import NORM
from modules.norm.layer_norm import NormConfig


@NORM.register_module
class LlamaRMSNorm(paddle.nn.Layer):
    config_class = NormConfig

    def __init__(self, config: NormConfig):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.config = config
        self.weight = paddle.create_parameter(
            shape=[config.n_embed],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=[config.n_embed]))
        )
        self.weight.stop_gradient = False

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        data_type = hidden_states.dtype
        hidden_states = hidden_states.cast(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.config.ln_eps)
        return self.weight * hidden_states.cast(data_type)
