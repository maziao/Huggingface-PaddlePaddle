import paddle.nn.functional as F
from dataclasses import dataclass
from config.base import BaseConfig
from modules.criterion import CRITERION


@dataclass
class CrossEntropyCriterionConfig(BaseConfig):
    pad_token_id: int = None


@CRITERION.register_module
class CrossEntropyCriterion:
    config_class = CrossEntropyCriterionConfig

    def __init__(self, config: CrossEntropyCriterionConfig):
        self.config = config

    def __call__(self, model, inputs):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])

        labels = inputs['labels'].reshape([-1])

        log_probs = F.log_softmax(
            outputs.logit,
            axis=-1
        ).reshape([-1, outputs.logit.shape[-1]])

        loss = F.nll_loss(
            input=log_probs,
            label=labels,
            ignore_index=self.config.pad_token_id,
            reduction='mean'
        )

        return loss, outputs
