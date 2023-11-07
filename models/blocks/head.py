import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from models.registry import build_activation, MODEL_HEAD


@dataclass
class TransformerLMHeadConfig(BaseConfig):
    n_vocab: int
    n_embed: int
    do_transform: bool = False
    act_fn: str = 'GELUActivation'
    ln_eps: float = 1e-05


@MODEL_HEAD.register_module
class TransformerLMHead(paddle.nn.Layer):
    config_class = TransformerLMHeadConfig

    def __init__(self, config: TransformerLMHeadConfig):
        super().__init__()
        self.config = config
        if config.do_transform:
            self.fc_trans = paddle.nn.Linear(in_features=config.n_embed, out_features=config.n_embed, bias_attr=True)
            self.act_fn = build_activation(config.act_fn)
            self.ln_trans = paddle.nn.LayerNorm(normalized_shape=config.n_embed, epsilon=config.ln_eps)
        self.fc_pred = paddle.nn.Linear(in_features=config.n_embed, out_features=config.n_vocab, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.config.do_transform:
            hidden_states = self.fc_trans(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = self.ln_trans(hidden_states)
        hidden_states = self.fc_pred(hidden_states)
        return hidden_states


@MODEL_HEAD.register_module
class TransformerClsHead(paddle.nn.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
