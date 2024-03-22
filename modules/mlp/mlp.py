import paddle
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from config.base import BaseConfig
from modules.mlp import MLP
from modules.activation import build_activation


@dataclass
class MLPConfig(BaseConfig):
    """
    Configuration class for MLPs in transformer-based models.

    Args:
        n_embed: token embedding dim, equal to hidden size in each layer.
        n_inner: hidden states dim in MLPs.
        act_fn_config: activation function used in MLPs.
        p_drop_mlp: dropout possibility in MLPs.
    """
    n_embed: int
    n_inner: int
    act_fn_config: Any
    p_drop_mlp: float = 0.0

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embed


@MLP.register_module
class TransformerMLP(paddle.nn.Layer):
    """
    Implementation of feed-forward network introduced in paper 'Attention is all you need'
    (https://arxiv.org/abs/1706.03762).

    Args:
        config: an instance of class `MLPConfig`.
    Usage:
        >>> import torch
        >>>
        >>> config = MLPConfig(n_embed=768, n_inner=3072, act_fn_config="relu", p_drop_mlp=0.1)
        >>> model = TransformerMLP(config)
        >>> a = torch.randn((8, 2048, 768), dtype=torch.float)
        >>> b = model(a)
        >>> b.size()
        torch.Size([8, 2048, 768])
    """
    config_class = MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.fc_in = paddle.nn.Linear(in_features=config.n_embed, out_features=config.n_inner)
        self.act_fn = build_activation(config.act_fn_config)
        self.fc_out = paddle.nn.Linear(in_features=config.n_inner, out_features=config.n_embed)
        self.mlp_dropout = paddle.nn.Dropout(p=config.p_drop_mlp)

    def forward(self, hidden_states: Optional[Tuple[paddle.Tensor]]) -> paddle.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.mlp_dropout(hidden_states)
        return hidden_states
