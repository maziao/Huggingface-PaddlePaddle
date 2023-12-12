import paddle
import paddle.nn.functional as F
from dataclasses import dataclass
from config.base import BaseConfig
from modules.criterion import CRITERION


@dataclass
class SimCTGCriterionConfig(BaseConfig):
    margin: float
    pad_token_id: int = None


@CRITERION.register_module
class SimCTGCriterion:
    config_class = SimCTGCriterionConfig

    def __init__(self, config: SimCTGCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)

        labels = inputs['labels']

        log_probs = F.log_softmax(outputs.logit, axis=-1)
        mle_loss = F.nll_loss(
            input=log_probs.reshape([-1, log_probs.shape[-1]]),
            label=labels.reshape([-1]),
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )
        last_hidden_states = outputs.hidden_states[-1]
        norm_rep = last_hidden_states / last_hidden_states.norm(axis=2, keepdim=True)
        perm = list(range(norm_rep.ndim))
        perm[1] = 2
        perm[2] = 1
        cosine_scores = paddle.matmul(x=norm_rep, y=norm_rep.transpose(perm=perm))
        cl_loss = compute_contrastive_loss(cosine_scores, self.config.margin)
        loss = mle_loss + cl_loss

        return loss, outputs


def compute_contrastive_loss(score_matrix, margin):
    """
       margin: predefined margin to push similarity score away
       score_matrix: batch_size x seq_len x seq_len; cosine similarity matrix
       input_ids: batch_size x seq_len
    """
    batch_size, seq_len, _ = score_matrix.shape
    gold_score = paddle.diagonal(x=score_matrix, offset=0, axis1=1, axis2=2)
    gold_score = paddle.unsqueeze(x=gold_score, axis=-1)
    assert gold_score.shape == list([batch_size, seq_len, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.shape == list([batch_size, seq_len, seq_len])
    loss_matrix = margin - difference_matrix
    loss_matrix = paddle.nn.functional.relu(x=loss_matrix)
    cl_loss = paddle.mean(x=loss_matrix)
    return cl_loss
