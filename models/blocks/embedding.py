import paddle
from dataclasses import dataclass
from typing import Optional
from config.base import BaseConfig
from models.registry import EMBEDDING


@dataclass
class TransformerEmbeddingConfig(BaseConfig):
    n_embed: int
    n_vocab: int
    n_pos: int = 0
    n_token_type: int = 0
    do_ln: bool = False
    ln_eps: float = 1e-05
    do_drop: bool = True
    p_drop_embed: float = 0.1


@EMBEDDING.register_module
class TransformerEmbedding(paddle.nn.Layer):
    config_class = TransformerEmbeddingConfig

    def __init__(self, config: TransformerEmbeddingConfig):
        super().__init__()
        self.config = config
        self.token_embed = paddle.nn.Embedding(num_embeddings=config.n_vocab, embedding_dim=config.n_embed)
        if config.n_pos != 0:
            self.pos_embed = paddle.nn.Embedding(num_embeddings=config.n_pos, embedding_dim=config.n_embed)
        if config.n_token_type != 0:
            self.type_embed = paddle.nn.Embedding(num_embeddings=config.n_token_type, embedding_dim=config.n_embed)
        if config.do_ln:
            self.ln = paddle.nn.LayerNorm(normalized_shape=config.n_embed, epsilon=config.ln_eps)
        if config.do_drop:
            self.embed_dropout = paddle.nn.Dropout(p=config.p_drop_embed)

    def forward(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            past_length: Optional[int] = 0,
            token_type_ids: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            input_embeds: Optional[paddle.Tensor] = None,
            concat_strategy: Optional[str] = 'id_first'
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
            concat_strategy: choose from ['id_first', 'embed_first']
        Return:
            final embedding tensor [batch_size, seq_len_total, n_embed]
        """
        if input_ids is not None and input_embeds is not None:
            len_input_seq = input_ids.shape[-1] + input_embeds.shape[-2]
        elif input_ids is not None:
            len_input_seq = input_ids.shape[-1]
        elif input_embeds is not None:
            len_input_seq = input_embeds.shape[-2]
        else:
            raise ValueError(
                f'You have to specify at least one of `input_ids` and `input_embeds`.'
            )

        # device = input_ids.place if input_ids is not None else input_embeds.place

        if input_ids is not None:
            token_embeds = self.token_embed(input_ids)
            if input_embeds is not None:
                if concat_strategy == 'id_first':
                    token_embeds = paddle.concat(x=[token_embeds, input_embeds], axis=-2)
                elif concat_strategy == 'embed_first':
                    token_embeds = paddle.concat(x=[input_embeds, token_embeds], axis=-2)
                else:
                    raise NotImplementedError(
                        f"Argument `concat_strategy` only support 2 legal options currently: 'id_first' and "
                        f"'embed_first'. Strategy '{concat_strategy}' has not been implemented yet."
                    )
        else:
            token_embeds = input_embeds
        hidden_states = token_embeds

        if hasattr(self, 'pos_embed'):
            if position_ids is not None:
                if position_ids.shape[-1] != len_input_seq:
                    raise ValueError(
                        f'The length of `position_ids`({int(position_ids.shape[-1])}) should be equal to the sum of '
                        f'lengths of `input_ids` and `input_embeds`({int(len_input_seq)}). If `position_ids` is not '
                        f'specified with certain specific rules, setting it to `None` may resolve this error.'
                    )
                position_ids = position_ids.reshape([-1, len_input_seq])
            else:
                position_ids = paddle.arange(start=past_length, end=len_input_seq + past_length, dtype='int64')
                position_ids = position_ids.unsqueeze(axis=0).reshape([-1, len_input_seq])
            hidden_states += self.pos_embed(position_ids)

        if token_type_ids is not None:
            if token_type_ids.shape[-1] != len_input_seq:
                raise ValueError(
                    f'The length of `token_type_ids`({int(token_type_ids.shape[-1])}) should be equal to the sum of '
                    f'lengths of `input_ids` and `input_embeds`({int(len_input_seq)}).'
                )
            token_type_ids = token_type_ids.reshape([-1, len_input_seq])
            if hasattr(self, 'type_embed'):
                hidden_states += self.type_embed(token_type_ids)
            else:
                hidden_states += self.token_embed(token_type_ids)

        if hasattr(self, 'ln'):
            hidden_states = self.ln(hidden_states)
        if hasattr(self, 'embed_dropout'):
            hidden_states = self.embed_dropout(hidden_states)
        return hidden_states


@EMBEDDING.register_module
class RotaryEmbedding(paddle.nn.Layer):
    config_class = TransformerEmbeddingConfig

    def __init__(self, config: TransformerEmbeddingConfig):
        super().__init__()
        self.config = config
