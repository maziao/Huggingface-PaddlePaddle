import paddle
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from config.base import BaseConfig
from modules.mlp import MLP
from modules.activation import build_activation


@dataclass
class LlamaMLPConfig(BaseConfig):
    """
    Configuration class for MLPs in transformer-based models.

    Args:
        n_embed: token embedding dim, equal to hidden size in each layer.
        n_inner: hidden states dim in MLPs.
        act_fn_config: activation function used in MLPs.
    """
    n_embed: int
    n_inner: int
    act_fn_config: Any


@MLP.register_module
class LlamaMLP(paddle.nn.Layer):
    config_class = LlamaMLPConfig

    def __init__(self, config: LlamaMLPConfig):
        super().__init__()
        self.fc_in_1 = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_inner,
            bias_attr=False
        )
        self.act_fn = build_activation(config.act_fn_config)
        self.fc_in_2 = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_inner,
            bias_attr=False
        )
        self.fc_out = paddle.nn.Linear(
            in_features=config.n_inner,
            out_features=config.n_embed,
            bias_attr=False
        )

    def forward(self, hidden_states: Optional[Tuple[paddle.Tensor]]) -> paddle.Tensor:
        hidden_states_1 = self.fc_in_1(hidden_states)
        hidden_states_1 = self.act_fn(hidden_states_1)
        hidden_states_2 = self.fc_in_2(hidden_states)
        hidden_states = hidden_states_1 * hidden_states_2
        hidden_states = self.fc_out(hidden_states)
        return hidden_states
