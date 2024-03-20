import math
import paddle
import logging
from typing import Any, Optional, Tuple
from modules.attention import ATTENTION
from modules.embedding import build_embedding
from modules.attention.llama_attn import LlamaAttentionConfig
from modules.attention.attention import MultiHeadKeyValueAttentionOutput

logger = logging.getLogger(__name__)


@ATTENTION.register_module
class GPTNeoXAttention(paddle.nn.Layer):
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
        self.rotary_embed = build_embedding(config.rope_config)
        
        self.inv_norm_factor = 1.0 / math.sqrt(config.head_size)
        
    def _attn(
            self,
            query: paddle.Tensor,
            key: paddle.Tensor,
            value: paddle.Tensor,
            attention_mask: paddle.Tensor = None,
            head_mask: paddle.Tensor = None,
            linear_bias: paddle.Tensor = None
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

    def _split_heads(self, tensor):
        """
        Splits 3 * n_embed into 3 n_head * head_size.

        Args:
            tensor: fused query, key, value tensor, size [batch_size, seq_len, 3 * n_embed]
        Return:
            query, key, value tensor, size [batch_size, n_head, seq_len, head_size]
        """
        new_shape = tensor.shape[:-1] + [self.config.n_head, 3 * self.config.head_size]
        tensor = tensor.reshape(new_shape)
        
        query = tensor[..., :self.config.head_size].transpose(perm=[0, 2, 1, 3])
        key = tensor[..., self.config.head_size: 2 * self.config.head_size].transpose(perm=[0, 2, 1, 3])
        value = tensor[..., 2 * self.config.head_size:].transpose(perm=[0, 2, 1, 3])
        return query, key, value

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
        fused_qkv = self.c_attn(hidden_states)

        query, key, value = self._split_heads(fused_qkv)

        kv_seq_len = key.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]

        query_rot = query[..., :self.config.rope_config.rotary_head_size]
        query_pass = query[..., self.config.rope_config.rotary_head_size:]
        key_rot = key[..., :self.config.rope_config.rotary_head_size]
        key_pass = key[..., self.config.rope_config.rotary_head_size:]
        
        query_rot, key_rot = self.rotary_embed(query_rot, key_rot, value, layer_past)
        query = paddle.concat([query_rot, query_pass], axis=-1)
        key = paddle.concat([key_rot, key_pass], axis=-1)
    
        if layer_past is not None:
            past_key, past_value = layer_past
            key = paddle.concat(x=(past_key, key), axis=-2)
            value = paddle.concat(x=(past_value, value), axis=-2)

        past_key_values = (key, value) if use_cache else None

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
        
        print(f"attn_output: {attn_output}")

        return MultiHeadKeyValueAttentionOutput(
            attn_output=attn_output,
            attn_weights=attn_weights if output_attentions else None,
            layer_present=past_key_values
        )
