import paddle
from dataclasses import dataclass
from typing import Optional, Tuple
from config.base import BaseConfig
from models.registry import ATTENTION


@dataclass
class AttentionConfig(BaseConfig):
    n_embed: int
    n_pos: int
    n_head: int
    head_size: int
    p_drop_attn: float
    p_drop_resid: float
    bias_attn: bool = True
    bias_proj: bool = True
    cross_attn: bool = False
    scale_dot_product: bool = True
    scale_layer_wise: bool = False


@dataclass
class MultiHeadKeyValueAttentionOutput:
    attn_output: paddle.Tensor
    attn_weights: Optional[paddle.Tensor] = None
    layer_present: Optional[Tuple[paddle.Tensor]] = None


@ATTENTION.register_module
class MultiHeadKeyValueAttention(paddle.nn.Layer):
    """
    Implementation of multi-head key-value attention block first introduced in paper 'Attention is all you need'
    (https://arxiv.org/abs/1706.03762). In this implementation, `torch.nn.Linear` module is used to calculate
    query / key / value matrices, and it is completely equivalent to the Conv1D layer defined by Alec Radford et al.
    for OpenAI GPT (and also used in GPT-2).
    """
    config_class = AttentionConfig

    def __init__(self, config: AttentionConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.register_buffer(
            name='bias',
            tensor=paddle.tril(
                paddle.ones(shape=(config.n_pos, config.n_pos), dtype='bool')).reshape([1, 1, config.n_pos, config.n_pos])
        )
        if config.cross_attn:
            self.q_attn = paddle.nn.Linear(
                in_features=config.n_embed,
                out_features=config.n_embed,
                bias_attr=config.bias_attn
            )
            self.c_attn = paddle.nn.Linear(
                in_features=config.n_embed,
                out_features=2 * config.n_embed,
                bias_attr=config.bias_attn
            )
        else:
            self.c_attn = paddle.nn.Linear(
                in_features=config.n_embed,
                out_features=3 * config.n_embed,
                bias_attr=config.bias_attn
            )
        self.c_proj = paddle.nn.Linear(
            in_features=config.n_embed,
            out_features=config.n_embed,
            bias_attr=config.bias_proj
        )
        self.attn_dropout = paddle.nn.Dropout(p=config.p_drop_attn)
        self.resid_dropout = paddle.nn.Dropout(p=config.p_drop_resid)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
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
        attn_weights = paddle.matmul(x=query, y=key.transpose(perm=[0, 1, 3, 2]))
        if self.config.scale_dot_product:
            attn_weights = attn_weights / paddle.full(
                shape=[],
                fill_value=value.shape[-1] ** 0.5,
                dtype=attn_weights.dtype
            )
        if self.config.scale_layer_wise:
            if self.layer_idx is not None:
                attn_weights = attn_weights / float(self.layer_idx + 1)
            else:
                print(
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
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
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

    def _split_heads(self, tensor):
        """
        Splits n_embed dim into n_head and head_size.

        Args:
            tensor: raw attn_weights, size [batch_size, seq_len, n_embed]
        Return:
            attn_weights with heads split, size [batch_size, n_head, seq_len, head_size]
        """
        new_shape = tensor.shape[:-1] + [self.config.n_head, self.config.head_size]
        tensor = tensor.reshape(new_shape)
        return tensor.transpose(perm=[0, 2, 1, 3])

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
            hidden_states: Optional[Tuple[paddle.Tensor]],
            layer_past: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
            attention_mask: Optional[paddle.Tensor] = None,
            head_mask: Optional[paddle.Tensor] = None,
            encoder_hidden_states: Optional[paddle.Tensor] = None,
            encoder_attention_mask: Optional[paddle.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False
    ) -> MultiHeadKeyValueAttentionOutput:
        if encoder_hidden_states is not None:
            assert self.config.cross_attn, (f'`forward` method in `MultiHeadKeyValueAttention` got non-null '
                                            f'`encoder_hidden_states` argument but `cross_attn` in `AttentionConfig` '
                                            f'was set to `False`. If cross attention is needed, please set it to '
                                            f'`True` instead.')
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.config.n_embed, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # print(self.c_attn(hidden_states))
            # query, key, value = self.c_attn(hidden_states).split(self.config.n_embed, dim=2)
            query, key, value = self.c_attn(hidden_states).split(3, axis=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if layer_past:
            past_key, past_value = layer_past
            key = paddle.concat(x=(past_key, key), axis=-2)
            value = paddle.concat(x=(past_value, value), axis=-2)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return MultiHeadKeyValueAttentionOutput(
            attn_output=attn_output,
            attn_weights=attn_weights if output_attentions else None,
            layer_present=(key, value) if use_cache else None
        )


@ATTENTION.register_module
class LlamaAttention(paddle.nn.Layer):
    config_class = AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
