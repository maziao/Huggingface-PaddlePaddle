import paddle
from models.losses import Criterion, CriterionConfig
from dataclasses import dataclass


@dataclass
class CandidatePenaltyCrossEntropyCriterionConfig(CriterionConfig):
    rank_alpha: float
    padding_idx: int = 1


class CandidatePenaltyCrossEntropyCriterion(Criterion):
    config_class = CandidatePenaltyCrossEntropyCriterionConfig
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, config: CandidatePenaltyCrossEntropyCriterionConfig):
        super().__init__(config)
        self.rank_alpha = config.rank_alpha
        self.padding_idx = config.padding_idx

    def forward(self, net_outputs, target, selected_matrix=None):
        nsentences = target.shape[0]
        target = target.reshape(-1)
        lprobs = paddle.nn.functional.softmax(x=net_outputs, axis=-1).log()
        lprobs = lprobs.reshape(-1, lprobs.shape[-1])
        true_token_lprobs = paddle.nn.functional.nll_loss(input=lprobs,
            label=target, ignore_index=self.padding_idx, reduction='none')
        mle_loss = true_token_lprobs.sum()
        chosen_tokens = (paddle.max(x=lprobs, axis=-1), paddle.argmax(x=
            lprobs, axis=-1))[1]
        gen_acc = (chosen_tokens.reshape(-1) == target.reshape(-1)).to('int64')
        valid_mask = (target != self.padding_idx).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        acc = valid_tokens.sum().item() / valid_mask.sum().item()
        if selected_matrix is not None:
            new_target = target.clone()
            selected_matrix = selected_matrix.reshape(-1)
            new_target[selected_matrix] = self.padding_idx
        else:
            new_target = target
        with paddle.no_grad():
            ctx_cands = new_target.unsqueeze(axis=0).expand(shape=[
                new_target.shape[0], new_target.shape[0]])
            ctx_cands_ = paddle.tril(ctx_cands, diagonal=-1) + self.padding_idx
            ctx_cands_ = ctx_cands_ * paddle.triu(ctx_cands_)
            ctx_cands = paddle.tril(ctx_cands, diagonal=-1) + ctx_cands_
            ctx_cands = paddle.where(ctx_cands == self.padding_idx ** 2,
                ctx_cands, self.padding_idx)
            negative_targets = paddle.zeros_like(x=lprobs).put_along_axis_(axis
                =1, indices=ctx_cands, values=1)
        one_minus_probs = paddle.clip(x=1.0 - lprobs.exp(), min=1e-05)
        custom_loss = -paddle.log(x=one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()
        loss = mle_loss + self.rank_alpha * custom_loss
        return loss, acc
