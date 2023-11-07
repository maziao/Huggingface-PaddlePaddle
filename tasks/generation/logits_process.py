import paddle
import inspect
import math
from typing import Callable, Iterable, List, Optional, Tuple, Union
import numpy as np



class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.'
            )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f'{self.__class__} is an abstract class. Only classes inheriting this class can be called.'
            )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **
        kwargs) ->paddle.Tensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys
                    ())[2:]):
                    raise ValueError(
                        f'Make sure that all the required parameters: {list(function_args.keys())} for {processor.__class__} are passed to the logits processor.'
                        )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(
                f'`min_length` has to be a non-negative integer, but is {min_length}'
                )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all([isinstance(i, int) for i in eos_token_id]) or any([(i <
            0) for i in eos_token_id]):
            pass
            # logger.warning(
            #     f'`eos_token_id` has to be a list of positive integers, but is {eos_token_id}'
            #     )
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, (i)] = -float('inf')
        return scores


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, prompt_length_to_skip: int, min_new_tokens: int,
        eos_token_id: Union[int, List[int]]):
        for arg_name, arg_value in [('prompt_length_to_skip',
            prompt_length_to_skip), ('min_new_tokens', min_new_tokens)]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(
                    f'`{arg_name}` has to be a positive integer, but is {arg_value}'
                    )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all([isinstance(i, int) for i in eos_token_id]) or any([(i <
            0) for i in eos_token_id]):
            pass
            # logger.warning(
            #     f'`eos_token_id` has to be a list of positive integers, but is {eos_token_id}'
            #     )
        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        if new_tokens_length < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, (i)] = -float('inf')
        return scores


class TemperatureLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not temperature > 0:
            raise ValueError(
                f'`temperature` has to be a strictly positive float, but is {temperature}'
                )
        self.temperature = temperature

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        scores = scores / self.temperature
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(
                f'`penalty` has to be a strictly positive float, but is {penalty}'
                )
        self.penalty = penalty

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        score = paddle.take_along_axis(arr=scores, axis=1, indices=input_ids)
        score = paddle.where(condition=score < 0, x=score * self.penalty, y
            =score / self.penalty)
        scores.put_along_axis_(axis=1, indices=input_ids, values=score)
        return scores


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] enforcing an exponential penalty on tokens that are not in the original input.

    Args:
        hallucination_penalty (`float`):
            The parameter for hallucination penalty. 1.0 means no penalty.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """

    def __init__(self, penalty: float, encoder_input_ids: paddle.Tensor):
        if not isinstance(penalty, float) or not penalty > 0:
            raise ValueError(
                f'`penalty` has to be a strictly positive float, but is {penalty}'
                )
        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        score = paddle.take_along_axis(arr=scores, axis=1, indices=self.
            encoder_input_ids)
        score = paddle.where(condition=score < 0, x=score * self.penalty, y
            =score / self.penalty)
        scores.put_along_axis_(axis=1, indices=self.encoder_input_ids,
            values=score)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float=-float('Inf'),
        min_tokens_to_keep: int=1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(
                f'`top_p` has to be a float > 0 and < 1, but is {top_p}')
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        sorted_logits, sorted_indices = paddle.sort(descending=False, x=scores
            ), paddle.argsort(descending=False, x=scores)
        cumulative_probs = paddle.nn.functional.softmax(sorted_logits, axis=-1
            ).cumsum(axis=-1)
        sorted_indices_to_remove = cumulative_probs <= 1 - self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[(...), -self.min_tokens_to_keep:] = 0
        indices_to_remove = sorted_indices_to_remove.put_along_axis(axis=1,
            indices=sorted_indices, values=sorted_indices_to_remove)
        scores = paddle.where(indices_to_remove, scores, self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float=-float('Inf'),
        min_tokens_to_keep: int=1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f'`top_k` has to be a strictly positive integer, but is {top_k}'
                )
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        top_k = min(self.top_k, scores.shape[-1])
        indices_to_remove = scores < paddle.topk(k=top_k, x=scores)[0][...,
            -1, None]
        scores = paddle.where(indices_to_remove, scores, self.filter_value)
        return scores


class TypicalLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs typical decoding. See [Typical Decoding for Natural Language
    Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, mass: float=0.9, filter_value: float=-float('Inf'),
        min_tokens_to_keep: int=1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(
                f'`typical_p` has to be a float > 0 and < 1, but is {mass}')
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        normalized = paddle.nn.functional.log_softmax(x=scores, axis=-1)
        p = paddle.exp(x=normalized)
        ent = -(normalized * p).nansum(axis=-1, keepdim=True)
        shifted_scores = paddle.abs(x=-normalized - ent)
        sorted_scores, sorted_indices = paddle.sort(descending=False, x=
            shifted_scores), paddle.argsort(descending=False, x=shifted_scores)
        sorted_logits = scores.take_along_axis(axis=-1, indices=sorted_indices)
        cumulative_probs = paddle.nn.functional.softmax(sorted_logits, axis=-1
            ).cumsum(axis=-1)
        last_ind = (cumulative_probs < self.mass).sum(axis=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = (sorted_scores > sorted_scores.take_along_axis(axis=1, indices=last_ind.reshape([-1, 1])))
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[(...), :self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.put_along_axis(axis=1,
            indices=sorted_indices, values=sorted_indices_to_remove)
        scores = paddle.where(indices_to_remove, scores, self.filter_value)
        return scores


class EpsilonLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, epsilon: float, filter_value: float=-float('Inf'),
        min_tokens_to_keep: int=1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(
                f'`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}'
                )
        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f'`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}'
                )
        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        probabilities = paddle.nn.functional.softmax(scores, axis=-1)
        indices_to_remove = probabilities < self.epsilon
        top_k = min(self.min_tokens_to_keep, scores.shape[-1])
        indices_to_remove = indices_to_remove & (scores < paddle.topk(k=
            top_k, x=scores)[0][..., -1, None])
        scores = paddle.where(indices_to_remove, scores, self.filter_value)
        return scores


class EtaLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs eta-sampling, i.e. calculates a dynamic cutoff `eta := min(epsilon, sqrt(epsilon,
    e^-entropy(probabilities)))` and restricts to tokens with `prob >= eta`. Takes the largest min_tokens_to_keep
    tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered."""

    def __init__(self, epsilon: float, filter_value: float=-float('Inf'),
        min_tokens_to_keep: int=1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(
                f'`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}'
                )
        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f'`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}'
                )
        self.epsilon = paddle.to_tensor(data=epsilon)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        probabilities = paddle.nn.functional.softmax(scores, axis=-1)
        entropy = paddle.distribution.Categorical(logits=scores).entropy()
        eta = paddle.min(self.epsilon, paddle.sqrt(x=self.epsilon) *
            paddle.exp(x=-entropy))[..., None]
        indices_to_remove = probabilities < eta
        top_k = min(self.min_tokens_to_keep, scores.shape[-1])
        indices_to_remove = indices_to_remove & (scores < paddle.topk(k=
            top_k, x=scores)[0][..., -1, None])
        scores = paddle.where(indices_to_remove, scores, self.filter_value)
        return scores


def _get_ngrams(ngram_size: int, prev_input_ids: paddle.Tensor, num_hypos: int
    ):
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: paddle.
    Tensor, num_hypos: int, cur_len: int) ->List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [_get_generated_ngrams(generated_ngrams[hypo_idx],
        prev_input_ids[hypo_idx], ngram_size, cur_len) for hypo_idx in
        range(num_hypos)]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f'`ngram_size` has to be a strictly positive integer, but is {ngram_size}'
                )
        self.ngram_size = ngram_size

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size,
            input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')
        return scores


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces no repetition of encoder input ids n-grams for the decoder ids. See
    [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350).

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: paddle.
        Tensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f'`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}'
                )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(axis=0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size,
            encoder_input_ids, self.batch_size)

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = [_get_generated_ngrams(self.generated_ngrams[
            hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size,
            cur_len) for hypo_idx in range(num_hypos)]
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float('inf')
        return scores


class NoBadWordsLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be sampled.

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated. In order to get the token ids of the words
            that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
            add_special_tokens=False).input_ids`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[
        int, List[int]]):
        if not isinstance(bad_words_ids, List) or len(bad_words_ids) == 0:
            raise ValueError(
                f'`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.'
                )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in
            bad_words_ids):
            raise ValueError(
                f'`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.'
                )
        if any(any(not isinstance(token_id, (int, np.integer)) or token_id <
            0 for token_id in bad_word_ids) for bad_word_ids in bad_words_ids):
            raise ValueError(
                f'Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}.'
                )
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(filter(lambda bad_token_seq: all([(
            bad_token_seq != [i]) for i in eos_token_id]), bad_words_ids))
        self.bad_words_id_length_1 = []
        self.bad_words_id_length_greater_than_1 = []
        for word in bad_words_ids:
            if len(word) == 1:
                self.bad_words_id_length_1.append(word[0])
            else:
                self.bad_words_id_length_greater_than_1.append(word)
        self.static_bad_words_mask: Optional[paddle.Tensor] = None
        for banned_token_seq in self.bad_words_id_length_greater_than_1:
            if len(banned_token_seq) == 0:
                raise ValueError(
                    f'Banned words token sequences {bad_words_ids} cannot have an empty list'
                    )

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        if self.static_bad_words_mask is None and len(self.
            bad_words_id_length_1) > 0:
            self.static_bad_words_mask = self._calc_static_bad_word_mask(scores
                )
        dynamic_banned_tokens = self._calc_banned_bad_words_ids(input_ids.
            tolist())
        scores = self._set_scores_to_inf_for_banned_tokens(scores,
            dynamic_banned_tokens)
        return scores

    def _calc_static_bad_word_mask(self, scores: paddle.Tensor
        ) ->paddle.Tensor:
        static_bad_words_mask = paddle.zeros(shape=scores.shape[1])
        static_bad_words_mask[self.bad_words_id_length_1] = 1
        return static_bad_words_mask.unsqueeze(axis=0).to(scores.place).astype(
            dtype='bool')

    def _tokens_match(self, prev_tokens: List[int], tokens: List[int]) ->bool:
        if len(tokens) == 0:
            return True
        elif len(tokens) > len(prev_tokens):
            return False
        else:
            return prev_tokens[-len(tokens):] == tokens

    def _calc_banned_bad_words_ids(self, prev_input_ids: List[List[int]]
        ) ->Iterable[int]:
        banned_tokens = []
        for prev_input_ids_slice in prev_input_ids:
            banned_tokens_slice = []
            for banned_token_seq in self.bad_words_id_length_greater_than_1:
                if self._tokens_match(prev_input_ids_slice,
                    banned_token_seq[:-1]):
                    banned_tokens_slice.append(banned_token_seq[-1])
            banned_tokens.append(banned_tokens_slice)
        return banned_tokens

    def _set_scores_to_inf_for_banned_tokens(self, scores: paddle.Tensor,
        banned_tokens: List[List[int]]) ->paddle.Tensor:
        """
        Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be a
        list of list of banned tokens to ban in the format [[batch index, vocabulary position],...

        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
        """
        banned_mask_list = []
        for idx, batch_banned_tokens in enumerate(banned_tokens):
            for token in batch_banned_tokens:
                if token <= scores.shape[1]:
                    banned_mask_list.append([idx, token])
                else:
                    pass
                    # logger.error(
                    #     f'An invalid bad word ID is defined: {token}. This ID is not contained in the vocabulary, and is therefore ignored.'
                    #     )
        if not banned_mask_list and self.static_bad_words_mask is None:
            return scores
        else:
            if banned_mask_list:
                banned_mask = paddle.to_tensor(data=banned_mask_list, dtype
                    ='int64')
                indices = paddle.ones(shape=len(banned_mask))
                banned_mask = paddle.sparse.LongTensor(banned_mask.t(),
                    indices, scores.shape).to(scores.place).to_dense().astype(
                    dtype='bool')
                if self.static_bad_words_mask is not None:
                    banned_mask = paddle.bitwise_or(x=banned_mask, y=self.
                        static_bad_words_mask)
            else:
                banned_mask = self.static_bad_words_mask
            scores = paddle.where(banned_mask, scores, -float('inf'))
            return scores


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, paddle.
        Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        mask = paddle.full_like(x=scores, fill_value=-math.inf)
        for batch_id, beam_sent in enumerate(input_ids.reshape([-1, self._num_beams, input_ids.shape[-1]])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self.
                    _prefix_allowed_tokens_fn(batch_id, sent)] = 0
        return scores + mask


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces diverse beam search. Note that this logits processor is only effective for
    [`PreTrainedModel.group_beam_search`]. See [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
        num_beams (`int`):
            Number of beams used for group beam search. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more
            details.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """

    def __init__(self, diversity_penalty: float, num_beams: int,
        num_beam_groups: int):
        if not isinstance(diversity_penalty, float
            ) or not diversity_penalty > 0.0:
            raise ValueError(
                '`diversity_penalty` should be a float strictly larger than 0.'
                )
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError(
                '`num_beams` should be an integer strictly larger than 1.')
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError(
                '`num_beam_groups` should be an integer strictly larger than 1.'
                )
        if num_beam_groups > num_beams:
            raise ValueError(
                '`beam_groups` has to be smaller or equal to `num_beams`.')
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor,
        current_tokens: paddle.Tensor, beam_group_idx: int) ->paddle.Tensor:
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self.
            _num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]
        if group_start_idx == 0:
            return scores
        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[batch_idx * self.
                _num_beams:batch_idx * self._num_beams + group_start_idx]
            token_frequency = paddle.bincount(x=previous_group_tokens,
                minlength=vocab_size).to(scores.place)
            scores[batch_idx * group_size:(batch_idx + 1) * group_size
                ] -= self._diversity_penalty * token_frequency
        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            num_tokens = scores.shape[1]
            scores[:, ([i for i in range(num_tokens) if i != self.
                bos_token_id])] = -float('inf')
            scores[:, (self.bos_token_id)] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            num_tokens = scores.shape[1]
            scores[:, ([i for i in range(num_tokens) if i not in self.
                eos_token_id])] = -float('inf')
            for i in self.eos_token_id:
                scores[:, (i)] = 0
        return scores


class InfNanRemoveLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.
    """

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        scores[scores != scores] = 0.0
        scores[scores == float('inf')] = paddle.finfo(scores.dtype).max
        return scores


class ExponentialDecayLengthPenalty(LogitsProcessor):
    """
    [`LogitsProcessor`] that exponentially increases the score of the eos_token_id after regulation_start has been
    reached.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        input_ids_seq_length (`int`):
            The length of the input sequence.
    """

    def __init__(self, exponential_decay_length_penalty: Tuple[int, float],
        eos_token_id: Union[int, List[int]], input_ids_seq_length: int):
        self.regulation_start = exponential_decay_length_penalty[0
            ] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len > self.regulation_start:
            for i in self.eos_token_id:
                scores[:, (i)] = scores[:, (i)] * pow(self.
                    regulation_factor, cur_len - self.regulation_start)
        return scores


class LogitNormalization(LogitsProcessor, LogitsWarper):
    """
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor
        ) ->paddle.Tensor:
        scores = scores.log_softmax(dim=-1)
        return scores


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    """
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not
    sampled at the begining of the generation.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.begin_index:
            scores[:, (self.begin_suppress_tokens)] = -float('inf')
        return scores


class SuppressTokensLogitsProcessor(LogitsProcessor):
    """This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they
    are not sampled."""

    def __init__(self, suppress_tokens):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids, scores):
        scores[:, (self.suppress_tokens)] = -float('inf')
        return scores


class ForceTokensLogitsProcessor(LogitsProcessor):
    """This processor takes a list of pairs of integers which indicates a mapping from generation indices to token
    indices that will be forced before sampling. The processor will set their log probs to `inf` so that they are
    sampled at their corresponding index."""

    def __init__(self, force_token_map: List[List[int]]):
        self.force_token_map = dict(force_token_map)

    def __call__(self, input_ids, scores):
        generation_idx = input_ids.shape[-1]
        current_token = self.force_token_map.get(generation_idx, None)
        if current_token is not None:
            scores[:, :] = -float('inf')
            scores[:, (current_token)] = 0
        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    """
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """

    def __init__(self, generate_config):
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.begin_index = len(generate_config.forced_decoder_ids) + 2
        if generate_config.forced_decoder_ids[-1][1
            ] == self.no_timestamps_token_id:
            self.begin_index -= 1
        self.max_initial_timestamp_index = (generate_config.
            max_initial_timestamp_index)

    def __call__(self, input_ids, scores):
        scores[:, (self.no_timestamps_token_id)] = -float('inf')
        if input_ids.shape[1] == self.begin_index - 1:
            scores[:, :] = -float('inf')
            scores[:, (self.timestamp_begin)] = 0
            return scores
        for k in range(input_ids.shape[0]):
            seq = list(input_ids[(k), self.begin_index:].tolist())
            last_was_timestamp = len(seq) >= 1 and seq[-1
                ] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2
                ] >= self.timestamp_begin
            if last_was_timestamp:
                if penultimate_was_timestamp:
                    scores[(k), self.timestamp_begin:] = -float('inf')
                else:
                    scores[(k), :self.eos_token_id] = -float('inf')
            if (input_ids.shape[1] == self.begin_index and self.
                max_initial_timestamp_index is not None):
                last_allowed = (self.timestamp_begin + self.
                    max_initial_timestamp_index)
                scores[:, last_allowed + 1:] = -float('inf')
        logprobs = paddle.nn.functional.log_softmax(x=scores.astype(dtype=
            'float32'), axis=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[(k), self.timestamp_begin:].logsumexp(
                axis=-1)
            max_text_token_logprob = logprobs[(k), :self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                scores[(k), :self.timestamp_begin] = -float('inf')
        return scores
