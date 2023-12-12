import paddle
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from config.base import BaseConfig
from modules.block import DECODER_BLOCK
from modules.mlp import build_mlp
from modules.attention import build_attention
from modules.norm import build_norm


@dataclass
class TransformerDecoderBlockConfig(BaseConfig):
    attn_config: Any
    mlp_config: Any
    ln_config: Any
    n_embed: int
    post_norm: bool = False
    add_cross_attn: bool = False


@dataclass
class TransformerDecoderBlockOutput:
    hidden_states: paddle.Tensor
    attn_weights: Optional[paddle.Tensor] = None
    layer_present: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None


@DECODER_BLOCK.register_module
class TransformerDecoderBlock(paddle.nn.Layer):
    config_class = TransformerDecoderBlockConfig

    def __init__(self, config: TransformerDecoderBlockConfig):
        super().__init__()
        self.config = config
        self.ln_1 = build_norm(config=config.ln_config)
        self.self_attn = build_attention(config=config.attn_config)
        if config.add_cross_attn:
            self.ln_cross_attn = build_norm(config=config.ln_config)
            self.cross_attn = build_attention(config=config.attn_config)
        self.ln_2 = build_norm(config=config.ln_config)
        self.mlp = build_mlp(config=config.mlp_config)

    def forward(
            self,
            hidden_states: Optional[Tuple[paddle.Tensor]],
            layer_past: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            head_mask: Optional[paddle.Tensor] = None,
            linear_bias: Optional[paddle.Tensor] = None,
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            encoder_attention_mask: Optional[paddle.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False
    ):
        residual = hidden_states

        if not self.config.post_norm:
            hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            linear_bias=linear_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs.attn_output
        hidden_states = attn_output + residual

        if self.config.post_norm:
            hidden_states = self.ln_1(hidden_states)

        if encoder_hidden_states is not None:
            if not hasattr(self, 'cross_attn'):
                raise ValueError(
                    f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention '
                    f'layers by setting `config.add_cross_attn=True`'
                )
            residual = hidden_states

            if not self.config.post_norm:
                hidden_states = self.ln_cross_attn(hidden_states)

            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            attn_output = cross_attn_outputs.attn_output
            hidden_states = residual + attn_output

            if self.config.post_norm:
                hidden_states = self.ln_cross_attn(hidden_states)

        residual = hidden_states

        if not self.config.post_norm:
            hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        if self.config.post_norm:
            hidden_states = self.ln_2(hidden_states)

        return TransformerDecoderBlockOutput(
            hidden_states=hidden_states,
            attn_weights=attn_outputs.attn_weights if output_attentions else None,
            layer_present=attn_outputs.layer_present if use_cache else None
        )
