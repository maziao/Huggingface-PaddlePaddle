import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from modules.norm import NORM


@dataclass
class NormConfig(BaseConfig):
    n_embed: int
    ln_eps: float = 1.0e-5


@NORM.register_module
class LayerNorm(paddle.nn.LayerNorm):
    config_class = NormConfig

    def __init__(self, config: NormConfig):
        super().__init__(normalized_shape=config.n_embed, epsilon=config.ln_eps)
