import paddle
from dataclasses import dataclass
from typing import Optional, Any
from config.base import BaseConfig
from modules.block import EMBEDDING_BLOCK
from modules.embedding import build_embedding
from modules.norm import build_norm


@dataclass
class TransformerEmbeddingBlockConfig(BaseConfig):
    token_embed_config: Any
    pos_embed_config: Any = None
    type_embed_config: Any = None
    ln_config: Any = None
    p_drop_embed: float = 0.1
    concat_strategy: Optional[str] = 'id_first'


@EMBEDDING_BLOCK.register_module
class TransformerEmbeddingBlock(paddle.nn.Layer):
    config_class = TransformerEmbeddingBlockConfig

    def __init__(self, config: TransformerEmbeddingBlockConfig):
        super().__init__()
        self.config = config
        self.token_embed = build_embedding(config.token_embed_config)
        if config.pos_embed_config is not None:
            self.pos_embed = build_embedding(config.pos_embed_config)
        if config.type_embed_config is not None:
            self.type_embed = build_embedding(config.type_embed_config)
        if config.ln_config is not None:
            self.ln = build_norm(config.ln_config)
        if config.p_drop_embed > 0.0:
            self.embed_dropout = paddle.nn.Dropout(p=config.p_drop_embed)

    @staticmethod
    def _get_seq_length(
            input_ids: Optional[paddle.Tensor],
            input_embeds: Optional[paddle.Tensor]
    ):
        if input_ids is not None and input_embeds is not None:
            seq_length = input_ids.shape[-1] + input_embeds.shape[-2]
        elif input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif input_embeds is not None:
            seq_length = input_embeds.shape[-2]
        else:
            raise ValueError(
                f'You have to specify at least one of `input_ids` and `input_embeds`.'
            )
        return seq_length

    def _get_token_embed(
            self,
            input_ids: Optional[paddle.Tensor],
            input_embeds: Optional[paddle.Tensor]
    ):
        if input_ids is not None:
            token_embeds = self.token_embed(input_ids)
            if input_embeds is not None:
                if self.config.concat_strategy == 'id_first':
                    token_embeds = paddle.concat(x=[token_embeds, input_embeds], axis=-2)
                elif self.config.concat_strategy == 'embed_first':
                    token_embeds = paddle.concat(x=[input_embeds, token_embeds], axis=-2)
                else:
                    raise NotImplementedError(
                        f"Argument `concat_strategy` only support 2 legal options currently: 'id_first' and "
                        f"'embed_first'. Strategy '{self.config.concat_strategy}' has not been implemented yet."
                    )
        else:
            token_embeds = input_embeds
        return token_embeds

    def _get_pos_embed(self, position_ids: Optional[paddle.Tensor], seq_length: int, past_length: int = 0):
        if hasattr(self, 'pos_embed'):
            if position_ids is not None:
                if position_ids.shape[-1] != seq_length:
                    raise ValueError(
                        f'The length of `position_ids`({int(position_ids.shape[-1])}) should be equal to the sum of '
                        f'lengths of `input_ids` and `input_embeds`({int(seq_length)}). If `position_ids` is not '
                        f'specified with certain specific rules, setting it to `None` may resolve this error.'
                    )
                position_ids = position_ids.reshape([-1, seq_length])
            else:
                position_ids = paddle.arange(start=past_length, end=seq_length + past_length, dtype='int64')
                position_ids = position_ids.unsqueeze(axis=0).reshape([-1, seq_length])
            return self.pos_embed(position_ids)
        else:
            return None

    def _get_type_embed(self, token_type_ids: Optional[paddle.Tensor], seq_length: int):
        if token_type_ids is not None:
            if token_type_ids.shape[-1] != seq_length:
                raise ValueError(
                    f'The length of `token_type_ids`({int(token_type_ids.shape[-1])}) should be equal to the sum of '
                    f'lengths of `input_ids` and `input_embeds`({int(seq_length)}).'
                )
            token_type_ids = token_type_ids.reshape([-1, seq_length])
            if hasattr(self, 'type_embed'):
                return self.type_embed(token_type_ids)
            else:
                return self.token_embed(token_type_ids)
        else:
            return None

    def forward(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            past_length: Optional[int] = 0,
            token_type_ids: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            input_embeds: Optional[paddle.Tensor] = None,
            **kwargs
    ) -> paddle.Tensor:
        """Forward method of `class TransformerEmbedding`. Input token ids or embedding tensors (or both, with the same
        `batch_size` and `n_embed`), this method will transform ids to embedding tensors first (if provided) and
        concatenate the result tensor and `input_embeds` tensor (in `seq_len` dim).

        Define seq_len_total = input_ids.size()[-1] + input_embeds.size()[-2]

        Parameters:
            input_ids: size [batch_size, seq_len]
            past_length: seq_len of past_key_values, defaults to 0. If specified, position embedding id will start
                         from `past_length` (thus it should always be no less than 0).
            token_type_ids: size [batch_size, seq_len_total]
            position_ids: size [batch_size, seq_len_total]
            input_embeds: size [batch_size, seq_len_embed, n_embed], will be concatenated with the embedded input_ids
        Return:
            final embedding tensor [batch_size, seq_len_total, n_embed]
        """
        seq_length = self._get_seq_length(input_ids, input_embeds)

        hidden_states = self._get_token_embed(input_ids, input_embeds)

        pos_embed = self._get_pos_embed(position_ids, seq_length, past_length)
        if pos_embed is not None:
            hidden_states += pos_embed

        token_type_embed = self._get_type_embed(token_type_ids, seq_length)
        if token_type_embed is not None:
            hidden_states += token_type_embed

        if hasattr(self, 'ln'):
            hidden_states = self.ln(hidden_states)
        if hasattr(self, 'embed_dropout'):
            hidden_states = self.embed_dropout(hidden_states)
        return hidden_states
