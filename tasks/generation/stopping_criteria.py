import paddle
import time
import warnings
from abc import ABC
from copy import deepcopy
from typing import Optional

STOPPING_CRITERIA_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs:
            Additional stopping criteria specific kwargs.

    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->bool:
        raise NotImplementedError('StoppingCriteria needs to be subclassed')


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->bool:
        return input_ids.shape[-1] >= self.max_length


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            f'The class `MaxNewTokensCriteria` is deprecated. Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` with `max_length = start_length + max_new_tokens` instead.'
            , FutureWarning)
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->bool:
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float]=None
        ):
        self.max_time = max_time
        self.initial_timestamp = time.time(
            ) if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->bool:
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) ->Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList,
    max_length: int) ->StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn(
            'You set different `max_length` for stopping criteria and `max_length` parameter'
            , UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
