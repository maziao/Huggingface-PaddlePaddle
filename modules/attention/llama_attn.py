import paddle
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from config.base import BaseConfig
from modules.attention import ATTENTION
from modules.embedding import build_embedding
from modules.attention.attention import MultiHeadKeyValueAttentionOutput

logger = logging.getLogger(__name__)


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch_size,
    n_key_value_head, seqlen, head_size) to (batch_size, num_attention_heads, seqlen, head_size)
    """
    batch_size, n_key_value_head, seq_len, head_size = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch_size, n_key_value_head, n_rep, seq_len, head_size])
    return hidden_states.reshape([batch_size, n_key_value_head * n_rep, seq_len, head_size])


@dataclass
class LlamaAttentionConfig(BaseConfig):
    n_embed: int
    n_pos: int
    n_head: int
    n_key_value_head: int
    head_size: int
    p_drop_attn: float
    p_drop_resid: float
    bias_attn: bool = False
    bias_proj: bool = False
    cross_attn: bool = False
    scale_dot_product: bool = True
    scale_layer_wise: bool = False
    layer_idx: int = None
    rope_config: Any = None


@ATTENTION.register_module
class LlamaAttention(paddle.nn.Layer):
    config_class = LlamaAttentionConfig

    def __init__(self, config: LlamaAttentionConfig):
        super().__init__()
        self.config = config
        self.bias = paddle.tril(
            paddle.ones(
                shape=(config.n_pos, config.n_pos),
                dtype='bool'
            )
        ).reshape([1, 1, config.n_pos, config.n_pos])
        self.q_proj = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_head * config.head_size,
            bias_attr=config.bias_attn
        )
        self.k_proj = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_key_value_head * config.head_size,
            bias_attr=config.bias_attn
        )
        self.v_proj = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_key_value_head * config.head_size,
            bias_attr=config.bias_attn
        )
        self.c_proj = paddle.nn.Linear(
            in_features=config.n_head * config.head_size,
            out_features=config.n_embed,
            bias_attr=config.bias_proj
        )
        self.attn_dropout = paddle.nn.Dropout(p=config.p_drop_attn)
        self.resid_dropout = paddle.nn.Dropout(p=config.p_drop_resid)
        self.rotary_embed = build_embedding(config.rope_config)

    def _attn(
            self,
            query: paddle.Tensor,
            key: paddle.Tensor,
            value: paddle.Tensor,
            attention_mask: paddle.Tensor = None,
            head_mask: paddle.Tensor = None
    ):
        """
        Attention operation in an attention block, including attention masking and head masking.

        Args:
            query: query tensor with head being split, size [batch_size, n_head, seq_len, head_size]
            key: similar with query
            value: similar with query
            attention_mask: optional, additional mask which will be added to attn_weights after causal masking is
                            performed (in self-attention).
            head_mask: optional, mask certain heads of attn_weights.

        Return:
            (attn_output, attn_weights):
        """

        # Basic attention weights calculation
        attn_weights = paddle.matmul(x=query, y=key.transpose(perm=[0, 1, 3, 2]))

        # Perform scale dot product
        if self.config.scale_dot_product:
            attn_weights = attn_weights / paddle.full(
                shape=[],
                fill_value=value.shape[-1] ** 0.5,
                dtype=attn_weights.dtype
            )

        # Scale attention weights by layer id
        if self.config.scale_layer_wise:
            if self.config.layer_idx is not None:
                attn_weights = attn_weights / float(self.config.layer_idx + 1)
            else:
                logger.warn(
                    f'Argument `scale_layer_wise` has been set `True` in `AttentionConfig`, while `layer_idx` was not '
                    f'provided when initializing `MultiHeadKeyValueAttention` block. Thus layer-wise scaling will not '
                    f'be performed. If it is truly needed, please provide `layer_idx` when initializing '
                    f'`MultiHeadKeyValueAttention` block instead.'
                )

        """
        Perform causal mask in self-attention to prevent prefix from being aware of the follow-up context.
            Example: 
            query_length = 5, key_length = 7, attn_weights [7, 5]
            masked attn_weights:
                   0    1    2    3    4    5    6
              0                 -inf -inf -inf -inf
              1                      -inf -inf -inf
              2                           -inf -inf
              3                                -inf
              4
        NOTE: query_length should be no greater than key_length. Typically, query and key share the same length.
        """
        if not self.config.cross_attn:
            query_length, key_length = query.shape[-2], key.shape[-2]
            if self.config.n_pos != 0:
                causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
            else:
                causal_mask = paddle.tril(
                    paddle.ones(
                        shape=attn_weights.shape[-2:],
                        dtype='bool'
                    )
                ).reshape([1, 1] + attn_weights.shape[-2:])
            mask_value = paddle.finfo(attn_weights.dtype).min
            mask_value = paddle.full(shape=[], fill_value=mask_value, dtype=attn_weights.dtype)
            attn_weights = paddle.where(condition=causal_mask, x=attn_weights, y=mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1)
        attn_weights = attn_weights.astype(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = paddle.matmul(x=attn_weights, y=value)

        return attn_output, attn_weights

    def _merge_heads(self, tensor):
        """
        Merges n_head dim and head_size dim into n_embed. (Reversed operation of `_split_heads`)

        Args:
            tensor: raw attn_output, size [batch_size, n_head, seq_len, head_size]
        Return:
            attn_output with heads merged, size [batch_size, seq_len, n_embed]
        """
        tensor = tensor.transpose(perm=[0, 2, 1, 3])
        new_shape = tensor.shape[:-2] + [self.config.n_embed]
        return tensor.reshape(new_shape)

    def forward(
            self,
            hidden_states: paddle.Tensor,
            layer_past: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            head_mask: Optional[paddle.Tensor] = None,
            linear_bias: Optional[paddle.Tensor] = None,
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            encoder_attention_mask: Optional[paddle.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query = (
            self.q_proj(hidden_states)
            .reshape([batch_size, seq_len, self.config.n_head, self.config.head_size])
            .transpose(perm=[0, 2, 1, 3])
        )
        key = (
            self.k_proj(hidden_states)
            .reshape([batch_size, seq_len, self.config.n_key_value_head, self.config.head_size])
            .transpose(perm=[0, 2, 1, 3])
        )
        value = (
            self.v_proj(hidden_states)
            .reshape([batch_size, seq_len, self.config.n_key_value_head, self.config.head_size])
            .transpose(perm=[0, 2, 1, 3])
        )

        kv_seq_len = key.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]

        query, key = self.rotary_embed(query, key, value, layer_past)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = paddle.concat(x=(past_key, key), axis=-2)
            value = paddle.concat(x=(past_value, value), axis=-2)

        key = repeat_kv(key, self.config.n_head // self.config.n_key_value_head)
        value = repeat_kv(value, self.config.n_head // self.config.n_key_value_head)

        attn_output, attn_weights = self._attn(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return MultiHeadKeyValueAttentionOutput(
            attn_output=attn_output,
            attn_weights=attn_weights if output_attentions else None,
            layer_present=(key, value) if use_cache else None
        )
