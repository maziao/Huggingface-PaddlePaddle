import random
import paddle
import paddle.nn.functional as F
from dataclasses import dataclass
from config.base import BaseConfig
from models.registry import CRITERION


@dataclass
class DITTOCriterionConfig(BaseConfig):
    sequence_tune_rate: float
    rep_reduce_gamma: float
    end_sentence_decoded: int = None
    pad_token_id: int = None


@CRITERION.register_module
class DITTOCriterion:
    config_class = DITTOCriterionConfig

    def __init__(self, config: DITTOCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        do_mle_step = True
        if paddle.rand(shape=[1]).item() < self.config.sequence_tune_rate:
            reorganized_inputs = reorganize_sentence(
                inputs['input_ids'],
                end_sentence_decoded=self.config.end_sentence_decoded
            )
            if reorganized_inputs is None:
                do_mle_step = True
            else:
                outputs = model(reorganized_inputs['input_ids'])
                loss = ditto_loss(reorganized_inputs, outputs, rep_reduce_gamma=self.config.rep_reduce_gamma)

        if do_mle_step:
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            labels = inputs['labels'].reshape([-1])
            log_probs = F.log_softmax(x=outputs.logit, axis=-1).reshape([-1, outputs.logit.shape[-1]])
            loss = F.nll_loss(
                input=log_probs,
                label=labels,
                ignore_index=self.config.pad_token_id,
                reduction='mean'
            )
        return loss, outputs


def reorganize_sentence(batched_input_ids, end_sentence_decoded: int):
    seq_len = batched_input_ids.shape[-1]
    prefix_len_list = []
    continuation_len_list = []
    reorganized_input_ids = []
    reorganized_labels = []
    for i, x in enumerate(batched_input_ids):
        xl = x.tolist()
        sentence_end_indices = []
        for idx, token in enumerate(xl):
            if token == end_sentence_decoded:
                sentence_end_indices.append(idx)
        try:
            sen_idx = random.randint(1, len(sentence_end_indices) - 2)
            last_sen_start = sentence_end_indices[sen_idx - 1] + 1
            sen_start = sentence_end_indices[sen_idx]
            sen_end = sentence_end_indices[sen_idx + 1]
        except ValueError:
            return None
        prefix = x[last_sen_start:sen_start]
        prefix_len = sen_start - last_sen_start
        left_tokens = seq_len - prefix_len
        continuation = x[sen_start:sen_end].reshape([1, -1])
        continuation_len = sen_end - sen_start
        repeat_time = left_tokens // continuation_len
        continuation = continuation.repeat(repeat_time + 1, 1).reshape([-1])
        new_sentence = paddle.concat(x=[prefix, continuation], axis=0)
        input_ids = new_sentence[:seq_len]
        labels = new_sentence[1:seq_len + 1]
        assert labels.shape[0] == seq_len
        prefix_len_list.append(prefix_len)
        continuation_len_list.append(continuation_len)
        reorganized_input_ids.append(input_ids)
        reorganized_labels.append(labels)
    reorganized_input_ids = paddle.stack(x=reorganized_input_ids)
    reorganized_labels = paddle.stack(x=reorganized_labels)
    return {
        'input_ids': reorganized_input_ids,
        'labels': reorganized_labels,
        'prefix_len': prefix_len_list,
        'continuation_len': continuation_len_list
    }


def ditto_loss(inputs, outputs, rep_reduce_gamma: float):
    batch_size, seq_len = inputs['input_ids'].shape
    labels = inputs['labels'].reshape(-1, 1)
    probs = F.softmax(x=outputs.logit, axis=-1).reshape([-1, outputs.logit.shape[-1]])
    target_probs = paddle.take_along_axis(
        arr=probs,
        axis=-1,
        indices=labels
    ).reshape([batch_size, seq_len])
    baseline_outputs = obtain_rep_baseline_prob(inputs, target_probs)
    one_minus_probs = paddle.clip(
        x=1.0 - paddle.abs(x=target_probs - baseline_outputs['probs'] * rep_reduce_gamma),
        min=1e-20
    )
    loss = -paddle.log(x=one_minus_probs) * baseline_outputs['mask']
    loss = loss.sum() / baseline_outputs['num_tokens']
    return loss


def obtain_rep_baseline_prob(inputs, target_probs):
    seq_len = inputs['input_ids'].shape[1]
    probs = []
    mask = []
    valid_tokens = 0
    for i, x in enumerate(target_probs):
        prefix_len = inputs['prefix_len'][i]
        continuation_len = inputs['continuation_len'][i]
        repeated_continuation_probs = paddle.zeros_like(x=x)
        repeated_continuation_probs[prefix_len + continuation_len:] = x[prefix_len:-continuation_len]
        repeated_continuation_mask = paddle.zeros_like(x=x, dtype='bool')
        repeated_continuation_mask[prefix_len + continuation_len:] = True
        probs.append(repeated_continuation_probs)
        mask.append(repeated_continuation_mask)
        valid_tokens += seq_len - prefix_len - continuation_len
    probs = paddle.stack(x=probs, axis=0)
    mask = paddle.stack(x=mask, axis=0)
    return {'probs': probs, 'mask': mask, 'num_tokens': valid_tokens}
