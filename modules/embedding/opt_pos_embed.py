import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from modules.embedding import EMBEDDING


@dataclass
class OPTLearnedPositionalEmbeddingConfig(BaseConfig):
    n_pos: int
    n_embed: int
    offset: int = 2


@EMBEDDING.register_module
class OPTLearnedPositionalEmbedding(paddle.nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    config_class = OPTLearnedPositionalEmbeddingConfig

    def __init__(self, config: OPTLearnedPositionalEmbeddingConfig):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.config = config
        super().__init__(num_embeddings=config.n_pos + config.offset, embedding_dim=config.n_embed)

    def forward(self, attention_mask: paddle.Tensor, past_length: int = 0):
        attention_mask = attention_mask.cast("int64")

        # create positions depending on attention_mask
        positions = (paddle.cumsum(attention_mask, axis=1).cast(attention_mask.dtype) * attention_mask).cast(
            "int64") - 1

        # cut positions if `past_length` is > 0
        positions = positions[:, past_length:]

        return super().forward(positions + self.config.offset)
