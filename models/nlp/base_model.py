import copy
import math

import paddle
from typing import Any, Optional, Tuple
from dataclasses import dataclass
from config.base import BaseConfig
from models.registry import ENCODER_ONLY_MODEL, DECODER_ONLY_MODEL, ENCODER_DECODER_MODEL, build_embedding, \
    build_encoder_block, build_decoder_block


@dataclass
class TransformerDecoderOnlyModelConfig(BaseConfig):
    embed_config: Any
    decoder_config: Any
    n_embed: int
    n_layer: int
    n_head: int = None
    do_ln: bool = True
    ln_eps: float = 1e-05
    perform_linear_bias: bool = False
    attn_window_size_loop_unit: list = None


@dataclass(frozen=True)
class TransformerBaseModelOutput:
    last_hidden_state: paddle.Tensor
    past_key_values: Optional[Tuple[Tuple[paddle.Tensor, paddle.Tensor]]]
    hidden_states: Optional[Tuple[paddle.Tensor]]
    attentions: Optional[Tuple[paddle.Tensor]]


@DECODER_ONLY_MODEL.register_module
class TransformerDecoderOnlyModel(paddle.nn.Layer):
    config_class = TransformerDecoderOnlyModelConfig

    def __init__(self, config: TransformerDecoderOnlyModelConfig):
        super().__init__()
        self.config = config
        decoder_layer_config_list = self.build_decoder_layer_config_list(config)
        self.embed = build_embedding(config=config.embed_config)
        self.decoder = paddle.nn.LayerList(
            sublayers=[build_decoder_block(decoder_layer_config_list[i]) for i in range(config.n_layer)]
        )
        if config.do_ln:
            self.ln_f = paddle.nn.LayerNorm(normalized_shape=config.n_embed, epsilon=config.ln_eps)

    @staticmethod
    def build_decoder_layer_config_list(config: TransformerDecoderOnlyModelConfig):
        decoder_config = [copy.deepcopy(config.decoder_config) for _ in range(config.n_layer)]
        if config.attn_window_size_loop_unit is not None:
            error_message = f"Attention window size should be (a list of) None or int, but got " \
                            f"{config.attn_window_size_loop_unit}"
            if isinstance(config.attn_window_size_loop_unit, int):
                loop_unit = [config.attn_window_size_loop_unit]
                loop_time = config.n_layer
            elif isinstance(config.attn_window_size_loop_unit, list):
                for window_size in config.attn_window_size_loop_unit:
                    assert window_size is None or isinstance(window_size, int), error_message
                loop_unit = config.attn_window_size_loop_unit
                loop_time = config.n_layer // len(config.attn_window_size_loop_unit) + 1
            else:
                raise ValueError(error_message)
            window_size_list = loop_unit * loop_time
            window_size_list = window_size_list[:config.n_layer]

            for i in range(config.n_layer):
                try:
                    decoder_config[i].attn_config.attn_window_size = window_size_list[i]
                except AttributeError:
                    continue

        for i in range(config.n_layer):
            try:
                decoder_config[i].attn_config.layer_idx = i
            except AttributeError:
                continue

        return decoder_config

    def build_alibi_tensor(self, attention_mask: paddle.Tensor) -> paddle.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
        Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
            attention_mask (`torch.Tensor`):
                Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
            num_heads (`int`, *required*):
                number of heads
            dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """
        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2 ** math.floor(math.log2(self.config.n_head))
        base = paddle.to_tensor(
            data=2 ** -2 ** -(math.log2(closest_power_of_2) - 3),
            dtype='float32',
            place=attention_mask.place
        )
        powers = paddle.arange(start=1, end=1 + closest_power_of_2, dtype='float32')
        slopes = paddle.pow(x=base, y=powers)
        if closest_power_of_2 != self.config.n_head:
            extra_base = paddle.to_tensor(
                data=2 ** -2 ** -(math.log2(2 * closest_power_of_2) - 3), dtype='float32',
                place=attention_mask.place
            )
            num_remaining_heads = min(closest_power_of_2, self.config.n_head - closest_power_of_2)
            extra_powers = paddle.arange(start=1, end=1 + 2 * num_remaining_heads, step=2, dtype='float32')
            slopes = paddle.concat(x=[slopes, paddle.pow(x=extra_base, y=extra_powers)], axis=0)
        arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        return alibi.reshape([batch_size * self.config.n_head, 1, seq_length])

    def forward(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            token_type_ids: Optional[paddle.Tensor] = None,
            input_embeds: Optional[paddle.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            head_mask: Optional[paddle.Tensor] = None,
            # additional args (only used in encoder-decoder models)
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            encoder_attention_mask: Optional[paddle.Tensor] = None,
            # output switches
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None
    ):
        if attention_mask is not None:
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            elif input_embeds is not None:
                batch_size = input_embeds.shape[0]
            else:
                raise ValueError(
                    'You have to specify either `input_ids` or `inputs_embeds`.'
                )
            if batch_size <= 0:
                raise ValueError('`batch_size` has to be defined and > 0')
            attention_mask = attention_mask.reshape([batch_size, -1])
            """
            We create a 3D attention mask from a 2D tensor mask.
            Sizes are [batch_size, 1, 1, to_seq_length].
            So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length].
            This attention mask is more simple than the triangular masking of causal attention
            used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            """
            attention_mask = attention_mask[:, None, None, :]

            """
            Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            masked positions, this operation will create a tensor which is 0.0 for
            positions we want to attend and the dtype's smallest value for masked positions.
            Since we are adding it to the raw scores before the softmax, this is
            effectively the same as removing these entirely.
            """
            attention_mask = attention_mask.cast(dtype='float32')
            attention_mask = (1.0 - attention_mask) * paddle.finfo(paddle.float32).min

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
        else:
            past_length = 0
            past_key_values = tuple([None] * self.config.n_layer)

        hidden_states = self.embed(
            input_ids=input_ids,
            past_length=past_length,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            input_embeds=input_embeds,
            concat_strategy='id_first'
        )

        if self.config.perform_linear_bias:
            if attention_mask is None:
                attention_mask = paddle.ones(hidden_states.shape[:2], dtype=paddle.float32)
            linear_bias = self.build_alibi_tensor(
                attention_mask=attention_mask.reshape([-1, attention_mask.shape[-1]])
            ).cast(dtype=hidden_states.dtype)
        else:
            linear_bias = None

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = None
        all_hidden_states = () if output_hidden_states else None
        if head_mask is None:
            head_mask = [None] * self.config.n_layer

        for i, (block, layer_past) in enumerate(zip(self.decoder, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                linear_bias=linear_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions
            )

            hidden_states = outputs.hidden_states

            if use_cache is True:
                presents = presents + (outputs.layer_present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs.attn_weights,)

        if self.config.do_ln:
            hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return TransformerBaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )
