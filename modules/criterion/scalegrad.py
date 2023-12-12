import numpy as np
import paddle
import paddle.nn.functional as F
from dataclasses import dataclass
from config.base import BaseConfig
from modules.criterion import CRITERION


@dataclass
class ScaleGradCriterionConfig(BaseConfig):
    gamma: float
    pad_token_id: int = None


@CRITERION.register_module
class ScaleGradCriterion:
    config_class = ScaleGradCriterionConfig

    def __init__(self, config: ScaleGradCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

        labels = inputs['labels']

        probs = F.softmax(outputs.logit, axis=-1)
        novel_mask = get_novel_mask(labels, outputs.logit.shape[-1])
        rep_mask = ~novel_mask

        new_probs = probs * novel_mask * self.config.gamma + probs * rep_mask + 1e-8
        new_probs = F.normalize(new_probs, p=1, axis=-1)
        log_probs = paddle.log(new_probs)
        loss = F.nll_loss(
            input=log_probs.reshape([-1, log_probs.shape[-1]]),
            label=labels.reshape([-1]),
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )

        return loss, outputs


def get_novel_mask(labels, vocab_size):
    batch_size, seq_len = labels.shape
    zeros = paddle.zeros([batch_size, seq_len, vocab_size])

    target_index = paddle.triu(
        labels.unsqueeze(1).expand(
            [batch_size, seq_len, seq_len]
        ).transpose([0, 2, 1])
    ).transpose([0, 2, 1])
    matrix = zeros.put_along_axis_(
        indices=target_index,
        values=paddle.Tensor(np.array(1.0, dtype=np.float32)),
        axis=2,
        reduce='add'
    )
    matrix[:, :, 0] = 0
    summ_true = paddle.Tensor(np.array(list(range(1, seq_len + 1)), dtype=np.float32)).unsqueeze(0)
    summ_now = paddle.sum(matrix, axis=-1)
    diff = summ_true - summ_now
    matrix[:, :, 0] = diff
    matrix = paddle.concat((paddle.zeros([batch_size, 1, vocab_size]), matrix[:, :-1, :]), 1)
    novel_mask = matrix < 1.

    return novel_mask
