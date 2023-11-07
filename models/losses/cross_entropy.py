import paddle
from models.registry import CRITERION


@CRITERION.register_module
class CrossEntropyCriterion(paddle.nn.Layer):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=logits.reshape([-1, logits.shape[-1]]),
            label=label,
            ignore_index=self.ignore_index
        )
        return loss.mean()
