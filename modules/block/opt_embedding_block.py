import paddle
from typing import Optional
from modules.block import EMBEDDING_BLOCK
from modules.block.embedding_block import TransformerEmbeddingBlockConfig, TransformerEmbeddingBlock


@EMBEDDING_BLOCK.register_module
class OPTEmbeddingBlock(TransformerEmbeddingBlock):
    config_class = TransformerEmbeddingBlockConfig

    def _get_pos_embed_opt(self, attention_mask_2d: Optional[paddle.Tensor], batch_size: int, seq_length: int, past_length: int = 0):
        if hasattr(self, 'pos_embed'):
            if attention_mask_2d is None:
                attention_mask_2d = paddle.ones([batch_size, seq_length + past_length])
            elif attention_mask_2d.shape[1] != seq_length + past_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask_2d.shape[1]}, but its length should be "
                    f"{seq_length + past_length} (sum of the lengths of current and past inputs)"
                )

            return self.pos_embed(attention_mask_2d, past_length)
        else:
            return None

    def forward(
            self,
            input_ids: Optional[paddle.Tensor] = None,
            past_length: Optional[int] = 0,
            token_type_ids: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            input_embeds: Optional[paddle.Tensor] = None,
            attention_mask_2d: Optional[paddle.Tensor] = None
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
            attention_mask_2d: size [batch_size, seq_len]
        Return:
            final embedding tensor [batch_size, seq_len_total, n_embed]
        """
        seq_length = self._get_seq_length(input_ids, input_embeds)

        hidden_states = self._get_token_embed(input_ids, input_embeds)

        pos_embed = self._get_pos_embed_opt(attention_mask_2d, input_ids.shape[0], seq_length, past_length)
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
