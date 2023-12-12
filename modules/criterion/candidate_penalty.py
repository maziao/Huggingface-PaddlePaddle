import numpy as np
import paddle
import paddle.nn.functional as F
from dataclasses import dataclass
from config.base import BaseConfig
from modules.criterion import CRITERION


@dataclass
class CandidatePenaltyCriterionConfig(BaseConfig):
    rank_alpha: float
    pad_token_id: int = None


@CRITERION.register_module
class CandidatePenaltyCriterion:
    config_class = CandidatePenaltyCriterionConfig

    def __init__(self, config: CandidatePenaltyCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

        labels = inputs['input_ids'][:, 1:].reshape([-1])

        log_probs = F.log_softmax(outputs.logit[:, :-1, :], axis=-1)

        log_probs = log_probs.reshape([-1, log_probs.shape[-1]])
        mle_loss = F.nll_loss(
            input=log_probs,
            label=labels,
            ignore_index=self.config.pad_token_id,
            reduction='sum'
        )

        with paddle.no_grad():
            ctx_cands = labels.unsqueeze(0).expand([1, labels.shape[0], labels.shape[0]])
            ctx_cands_ = (paddle.tril(ctx_cands, -1) + self.config.pad_token_id)
            ctx_cands_ = ctx_cands_ * paddle.triu(ctx_cands_)
            ctx_cands = paddle.tril(ctx_cands, -1) + ctx_cands_

            ctx_cands = paddle.where(ctx_cands != (self.config.pad_token_id ** 2), ctx_cands, self.config.pad_token_id)
            ctx_cands = ctx_cands.squeeze(axis=0)
            negative_targets = paddle.zeros_like(x=log_probs).put_along_axis_(
                indices=ctx_cands,
                values=paddle.Tensor(np.array(1.0, dtype=np.float32)),
                axis=1
            )

        one_minus_probs = paddle.clip((1.0 - log_probs.exp()), min=1e-5)

        custom_loss = -paddle.log(one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()

        loss = mle_loss + self.config.rank_alpha * custom_loss
        loss /= paddle.sum(labels != self.config.pad_token_id)
        return loss, outputs
