import paddle
from typing import Any
from dataclasses import dataclass
from config.base import BaseConfig
from modules.head import MODEL_HEAD
from modules.activation import build_activation
from modules.norm import build_norm

import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformerLMHeadConfig(BaseConfig):
    n_vocab: int
    n_embed: int
    perform_transform: bool = False
    act_fn_config: Any = None
    ln_config: Any = None

    def __post_init__(self):
        if self.perform_transform is True and (self.act_fn_config is None or self.ln_config is None):
            logger.warning(f"`act_fn_config` and `ln_config` must be provided if performing transform in lm_head, "
                           f"but got {self.act_fn_config} and {self.ln_config}. Set `perform_transform` to False for "
                           f"safe training.")


@MODEL_HEAD.register_module
class TransformerLMHead(paddle.nn.Layer):
    config_class = TransformerLMHeadConfig

    def __init__(self, config: TransformerLMHeadConfig):
        super().__init__()
        self.config = config
        if config.perform_transform:
            self.fc_trans = paddle.nn.Linear(in_features=config.n_embed, out_features=config.n_embed, bias_attr=True)
            self.act_fn = build_activation(config.act_fn_config)
            self.ln_trans = build_norm(config.ln_config)
        self.fc_pred = paddle.nn.Linear(in_features=config.n_embed, out_features=config.n_vocab, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.config.perform_transform:
            hidden_states = self.fc_trans(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = self.ln_trans(hidden_states)
        hidden_states = self.fc_pred(hidden_states)
        return hidden_states


@MODEL_HEAD.register_module
class TransformerClassificationHead(paddle.nn.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
