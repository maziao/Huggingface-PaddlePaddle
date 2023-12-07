import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from models.registry import CRITERION


def batch_input_sequence_by_prefix_length(input_ids, prefix_len):
    seq_len = input_ids.shape[1]
    if seq_len > prefix_len:
        new_seq_len = seq_len // prefix_len * prefix_len
        input_ids = input_ids[:, :new_seq_len]
        batch = input_ids.reshape([-1, prefix_len])
    else:
        batch = input_ids.reshape([-1, seq_len])
    return batch


def sample_sequence(model, input_ids, prefix_len, continuation_len):
    continuation_logits = []
    context = input_ids
    assert context.shape[1] == prefix_len or context.shape[1] == input_ids.shape[1]
    prev = context
    output = context
    past = None
    for i in range(continuation_len):
        model_outputs = model(prev, past_key_values=past, output_key_values=True)
        logits = model_outputs.logit[:, -1, :]
        past = model_outputs.past_key_values
        prev = logits.argmax(axis=1, keepdim=True)
        continuation_logits.append(logits)
        output = paddle.concat(x=(output, prev), axis=1)
    continuation_logits = paddle.stack(x=continuation_logits, axis=1)
    return output, continuation_logits


def ngram_repeat_mask(xs, n):
    mask = paddle.zeros_like(x=xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x) - n):
            ng = tuple(xl[j:j + n])
            if ng in seen:
                mask[(i), j:j + n] = 1
            seen.add(ng)
    return mask


@dataclass
class UnlikelihoodTrainingSequenceLevelCriterionConfig(BaseConfig):
    prefix_len: int
    continuation_len: int
    sequence_ngram_n: int
    sequence_tune_rate: float
    pad_token_id: int = None


@CRITERION.register_module
class UnlikelihoodTrainingSequenceLevelCriterion:
    config_class = UnlikelihoodTrainingSequenceLevelCriterionConfig

    def __init__(self, config: UnlikelihoodTrainingSequenceLevelCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        if paddle.rand(shape=[1]).item() < self.config.sequence_tune_rate:
            prefix_len = min(self.config.prefix_len, inputs['input_ids'].shape[1])
            input_ids = batch_input_sequence_by_prefix_length(
                input_ids=inputs['input_ids'],
                prefix_len=self.config.prefix_len
            )
            completions, continuation_logits = sample_sequence(
                model=model,
                input_ids=input_ids,
                prefix_len=self.config.prefix_len,
                continuation_len=self.config.continuation_len
            )
            pred_tokens = completions[:, prefix_len:]
            mask = ngram_repeat_mask(pred_tokens, self.config.sequence_ngram_n).astype(dtype=continuation_logits.dtype)
            log_probs = paddle.nn.functional.log_softmax(x=continuation_logits, axis=-1)
            pred_log_probs = log_probs.reshape([-1, log_probs.shape[2]]).take_along_axis(
                axis=1,
                indices=pred_tokens.reshape([-1, 1])
            )
            one_minus_probs = paddle.clip(
                x=1.0 - pred_log_probs.exp(),
                min=1e-20
            ).reshape([pred_tokens.shape[0], pred_tokens.shape[1]])
            loss = -paddle.log(x=one_minus_probs) * mask
            loss = loss.sum()
            loss = loss / pred_tokens.size
        else:
            labels = inputs['input_ids'][:, 1:].reshape([-1])
            log_probs = paddle.nn.functional.log_softmax(
                x=outputs.logit[:, :-1, :],
                axis=-1
            ).reshape([-1, outputs.logit.shape[-1]])
            loss = paddle.nn.functional.nll_loss(
                input=log_probs,
                label=labels,
                ignore_index=self.config.pad_token_id,
                reduction='none'
            )
            loss = loss.mean()
        return loss, outputs
