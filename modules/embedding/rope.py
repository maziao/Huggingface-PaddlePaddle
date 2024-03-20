import paddle
from dataclasses import dataclass
from typing import List, Optional
from config.base import BaseConfig
from modules.embedding import EMBEDDING

import logging.config

logger = logging.getLogger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return paddle.concat([-x2, x1], axis=-1)


@dataclass
class RotaryPositionEmbeddingConfig(BaseConfig):
    head_size: int
    n_pos: int
    base: int = 10000
    scaling_type: Optional[str] = None
    scaling_factor: float = None

    def __post_init__(self):
        scaling_types = [None, 'linear', 'ntk']
        if self.scaling_type not in scaling_types:
            logger.warn(f"Scaling type of RoPE should be one of {scaling_types}, but got {self.scaling_type}. Set it "
                        f"to None for safe training.")
            self.scaling_type = None

        if self.scaling_type is not None and (self.scaling_factor is None or self.scaling_factor <= 0.0):
            logger.warn(f"Scaling factor should be a positive float point number when scaling type is set to "
                        f"{self.scaling_type}, but got {self.scaling_factor}. Set scaling type to None for safe "
                        f"training.")
            self.scaling_type = None


@EMBEDDING.register_module
class RotaryPositionEmbedding(paddle.nn.Layer):
    # TODO: support customizable rotary dim
    config_class = RotaryPositionEmbeddingConfig

    def __init__(self, config: RotaryPositionEmbeddingConfig):
        super().__init__()
        self.config = config
        self.inv_freq = 1.0 / (self.config.base ** (
                    paddle.arange(0, self.config.head_size, 2).cast(paddle.float32) / self.config.head_size))
        self.max_seq_len_cached = None
        self._set_cos_sin_cache(seq_len=config.n_pos, dtype=paddle.get_default_dtype())

    def _set_inv_freq_ntk(self, seq_len: int):
        if seq_len > self.config.n_pos:
            base = self.config.base * (
                    (self.config.scaling_factor * seq_len / self.config.n_pos) - (self.config.scaling_factor - 1)
            ) ** (self.config.head_size / (self.config.head_size - 2))
            self.inv_freq = 1.0 / (base ** (
                    paddle.arange(0, self.config.head_size, 2).cast(paddle.float32) / self.config.head_size))

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if self.config.scaling_type == 'ntk':
            self._set_inv_freq_ntk(seq_len)

        t = paddle.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        if self.config.scaling_type == 'linear':
            t = t / self.config.scaling_factor

        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)

        emb = paddle.concat([freqs, freqs], axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :].cast(dtype)
        self.sin_cached = emb.sin()[None, None, :, :].cast(dtype)

    def forward(self, query, key, value, layer_past: List[paddle.Tensor] = None):
        seq_len = query.shape[-2]
        past_len = 0
        if layer_past is not None:
            past_len = layer_past[0].shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=past_len + seq_len, dtype=value.dtype)

        cos = self.cos_cached[:, :, :past_len + seq_len, ...].cast(dtype=value.dtype)
        sin = self.sin_cached[:, :, :past_len + seq_len, ...].cast(dtype=value.dtype)

        position_ids = paddle.arange(start=past_len, end=past_len + seq_len, dtype='int64')
        position_ids = position_ids.unsqueeze(axis=0).reshape([-1, seq_len])

        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed, k_embed


@dataclass
class MistralRotaryEmbeddingConfig(BaseConfig):
    rotary_head_size: int
    n_pos: int
    base: int = 10000
    scaling_type: Optional[str] = None
    scaling_factor: float = None

    def __post_init__(self):
        scaling_types = [None, 'linear', 'ntk']
        if self.scaling_type not in scaling_types:
            logger.warn(f"Scaling type of RoPE should be one of {scaling_types}, but got {self.scaling_type}. Set it "
                        f"to None for safe training.")
            self.scaling_type = None

        if self.scaling_type is not None and (self.scaling_factor is None or self.scaling_factor <= 0.0):
            logger.warn(f"Scaling factor should be a positive float point number when scaling type is set to "
                        f"{self.scaling_type}, but got {self.scaling_factor}. Set scaling type to None for safe "
                        f"training.")
            self.scaling_type = None

@EMBEDDING.register_module
class MistralRotaryEmbedding(paddle.nn.Layer):
    # TODO: support linear and ntk scaling
    config_class = MistralRotaryEmbeddingConfig
    
    def __init__(self, config: MistralRotaryEmbeddingConfig):
        super().__init__()

        self.dim = config.rotary_head_size
        self.max_position_embeddings = config.n_pos
        self.base = config.base
        inv_freq = 1.0 / (self.base ** (paddle.arange(0, self.dim, 2, dtype=paddle.int64).cast(paddle.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=config.n_pos, dtype=paddle.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(self.max_seq_len_cached, dtype=paddle.int64).cast(self.inv_freq.dtype)

        freqs = paddle.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat((freqs, freqs), axis=-1)
        self.register_buffer("cos_cached", emb.cos().cast(dtype))
        self.register_buffer("sin_cached", emb.sin().cast(dtype))

    def forward(self, query, key, value, layer_past: List[paddle.Tensor] = None):
        seq_len = query.shape[-2]
        past_len = 0
        if layer_past is not None:
            past_len = layer_past[0].shape[-2]

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=value.dtype)

        position_ids = paddle.arange(start=past_len, end=past_len + seq_len, dtype='int64')
        position_ids = position_ids.unsqueeze(axis=0).reshape([-1, seq_len])

        cos = self.cos_cached[:seq_len].cast(value.dtype)
        sin = self.sin_cached[:seq_len].cast(value.dtype)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q_embed = (query * cos) + (rotate_half(query) * sin)
        k_embed = (key * cos) + (rotate_half(key) * sin)
        return q_embed, k_embed
