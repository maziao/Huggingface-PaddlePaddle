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
    do_ln: bool = True
    ln_eps: float = 1e-05


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
        self.embed = build_embedding(config=config.embed_config)
        self.decoder = paddle.nn.LayerList(
            sublayers=[build_decoder_block(config.decoder_config) for _ in range(config.n_layer)]
        )
        if config.do_ln:
            self.ln_f = paddle.nn.LayerNorm(normalized_shape=config.n_embed, epsilon=config.ln_eps)

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
            attention_mask = attention_mask.to(dtype='float32')
            attention_mask = (1.0 - attention_mask) * paddle.finfo('float32').min

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

        if hasattr(self, 'ln_f'):
            hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return TransformerBaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )
