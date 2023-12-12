import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from modules.embedding import EMBEDDING


@dataclass
class TokenEmbeddingConfig(BaseConfig):
    n_embed: int
    n_vocab: int


@EMBEDDING.register_module
class TokenEmbedding(paddle.nn.Embedding):
    config_class = TokenEmbeddingConfig

    def __init__(self, config: TokenEmbeddingConfig):
        super().__init__(num_embeddings=config.n_vocab, embedding_dim=config.n_embed)


@dataclass
class PositionEmbeddingConfig(BaseConfig):
    n_embed: int
    n_pos: int


@EMBEDDING.register_module
class PositionEmbedding(paddle.nn.Embedding):
    config_class = PositionEmbeddingConfig

    def __init__(self, config: PositionEmbeddingConfig):
        super().__init__(num_embeddings=config.n_pos, embedding_dim=config.n_embed)
