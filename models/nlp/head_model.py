import os.path

import paddle
from typing import Any, Optional, Tuple
from dataclasses import dataclass
from config.base import BaseConfig
from models.registry import LM_HEAD_MODEL, build_decoder_only_model, build_model_head


@dataclass
class TransformerLMHeadModelConfig(BaseConfig):
    transformer_config: Any
    lm_head_config: Any


@dataclass
class TransformerLMHeadModelOutput:
    loss: Optional[paddle.Tensor] = None
    logit: paddle.Tensor = None
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    cross_attentions: Optional[Tuple[paddle.Tensor]] = None


@LM_HEAD_MODEL.register_module
class TransformerLMHeadModel(paddle.nn.Layer):
    config_class = TransformerLMHeadModelConfig

    def __init__(self, config: TransformerLMHeadModelConfig):
        super().__init__()
        self.transformer = build_decoder_only_model(config=config.transformer_config)
        self.lm_head = build_model_head(config=config.lm_head_config)

    def forward(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            token_type_ids: Optional[paddle.Tensor] = None,
            input_embeds: Optional[paddle.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            head_mask: Optional[paddle.Tensor] = None,
            cross_attention_head_mask: Optional[paddle.Tensor] = None,
            output_key_values: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            input_embeds=input_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=output_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        hidden_states = transformer_outputs.last_hidden_state
        lm_logit = self.lm_head(hidden_states)

        return TransformerLMHeadModelOutput(
            logit=lm_logit,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=None
        )

    def from_pretrained(self, pretrained_model_path: str):
        state_dict = paddle.load(os.path.join(pretrained_model_path, 'paddle_model.pdparams'))

        has_lm_head = False
        token_embed_key = None
        for key in state_dict.keys():
            split_key = key.split('.')
            if split_key[-1] == 'lm_head':
                has_lm_head = True
            if split_key[-2] == 'token_embed':
                token_embed_key = key

        assert token_embed_key is not None, f"Pretrained model does not have a token embed matrix."

        if not has_lm_head:
            state_dict['lm_head.fc_pred.weight'] = state_dict[token_embed_key].T

        self.set_state_dict(state_dict)
