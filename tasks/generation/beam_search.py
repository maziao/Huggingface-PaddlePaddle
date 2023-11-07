import paddle
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Tuple, Union
import numpy as np
from .beam_constraints import Constraint, ConstraintListState


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for [`~PreTrainedModel.beam_search`] and
    [`~PreTrainedModel.beam_sample`].
    """

    @abstractmethod
    def process(self, input_ids: paddle.Tensor, next_scores: paddle.Tensor,
                next_tokens: paddle.Tensor, next_indices: paddle.Tensor, **kwargs
                ) -> Tuple[paddle.Tensor]:
        raise NotImplementedError('This is an abstract method.')

    @abstractmethod
    def finalize(self, input_ids: paddle.Tensor, next_scores: paddle.Tensor,
                 next_tokens: paddle.Tensor, next_indices: paddle.Tensor, max_length:
            int, **kwargs) -> paddle.Tensor:
        raise NotImplementedError('This is an abstract method.')


class BeamSearchScorer(BeamScorer):
    """
    [`BeamScorer`] implementing standard beam search decoding.

    Adapted in part from [Facebook's XLM beam search
    code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

    Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan's DBS
    implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(self, batch_size: int, num_beams: int, device: str,
                 length_penalty: Optional[float] = 1.0, do_early_stopping: Optional[
                Union[bool, str]] = False, num_beam_hyps_to_keep: Optional[int] = 1,
                 num_beam_groups: Optional[int] = 1, max_length: Optional[int] = None):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self._is_init = False
        self._beam_hyps = [BeamHypotheses(num_beams=self.num_beams,
                                          length_penalty=self.length_penalty, early_stopping=self.
                                          do_early_stopping, max_length=max_length) for _ in range(
            batch_size)]
        self._done = paddle.to_tensor(data=[(False) for _ in range(
            batch_size)], dtype='bool', place=self.device)
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f'`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead.'
            )
        if not isinstance(num_beam_groups, int
                          ) or num_beam_groups > num_beams or num_beams % num_beam_groups != 0:
            raise ValueError(
                f'`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}.'
            )

    @property
    def is_done(self) -> bool:
        return self._done.astype('bool').all()

    def process(self, input_ids: paddle.Tensor, next_scores: paddle.Tensor,
                next_tokens: paddle.Tensor, next_indices: paddle.Tensor,
                pad_token_id: Optional[int] = None, eos_token_id: Optional[Union[int,
            List[int]]] = None, beam_indices: Optional[paddle.Tensor] = None) -> Tuple[
        paddle.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not batch_size == input_ids.shape[0] // self.group_size:
            if self.num_beam_groups > 1:
                raise ValueError(
                    f'A group beam size of {input_ids.shape[0]} is used as the input, but a group beam size of {self.group_size} is expected by the beam scorer.'
                )
            else:
                raise ValueError(
                    f'A beam size of {input_ids.shape[0]} is used as the input, but a beam size of {self.group_size} is expected by the beam scorer.'
                )
        device = input_ids.place
        next_beam_scores = paddle.zeros(shape=(batch_size, self.group_size),
                                        dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros(shape=(batch_size, self.group_size),
                                        dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros(shape=(batch_size, self.group_size
                                                ), dtype=next_indices.dtype)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(
                        f'Batch can only be done if at least {self.num_beams} beams have been generated'
                    )
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError(
                        'Generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                    )
                next_beam_scores[(batch_idx), :] = 0
                next_beam_tokens[(batch_idx), :] = pad_token_id
                next_beam_indices[(batch_idx), :] = 0
                continue
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index
                                  ) in enumerate(zip(next_tokens[batch_idx], next_scores[
                batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                if eos_token_id is not None and next_token.item(
                ) in eos_token_id:
                    is_beam_token_worse_than_top_num_beams = (
                            beam_token_rank >= self.group_size)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    if beam_indices is not None:
                        beam_index = beam_indices[batch_beam_idx]
                        beam_index = beam_index + (batch_beam_idx,)
                    else:
                        beam_index = None
                    beam_hyp.add(input_ids[batch_beam_idx].clone(),
                                 next_score.item(), beam_indices=beam_index)
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1
                if beam_idx == self.group_size:
                    break
            if beam_idx < self.group_size:
                raise ValueError(
                    f'At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected.'
                )
            cur_len += 1
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len)
            return UserDict({'next_beam_scores': next_beam_scores.reshape([-1]),
                             'next_beam_tokens': next_beam_tokens.reshape([-1]),
                             'next_beam_indices': next_beam_indices.reshape([-1])})

    def finalize(self, input_ids: paddle.Tensor, final_beam_scores: paddle.
                 Tensor, final_beam_tokens: paddle.Tensor, final_beam_indices:
    paddle.Tensor, max_length: int, pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[Union[int, List[int]]] = None, beam_indices:
            Optional[paddle.Tensor] = None) -> Tuple[paddle.Tensor]:
        batch_size = len(self._beam_hyps)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_index = beam_indices[batch_beam_idx
                ] if beam_indices is not None else None
                beam_hyp.add(final_tokens, final_score, beam_indices=beam_index
                             )
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_indices = []
        best_scores = paddle.zeros(shape=batch_size * self.
                                   num_beam_hyps_to_keep, dtype='float32')
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                best_index = best_hyp_tuple[2]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp
                                                                       )
                best.append(best_hyp)
                best_indices.append(best_index)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length
                           ) if max_length is not None else sent_lengths_max
        decoded: paddle.Tensor = input_ids.new(batch_size * self.
                                               num_beam_hyps_to_keep, sent_max_len)
        if len(best_indices) > 0 and best_indices[0] is not None:
            indices: paddle.Tensor = input_ids.new(batch_size * self.
                                                   num_beam_hyps_to_keep, sent_max_len)
        else:
            indices = None
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, '`pad_token_id` has to be defined'
            decoded.fill_(value=pad_token_id)
        if indices is not None:
            indices.fill_(value=-1)
        for i, (hypo, best_idx) in enumerate(zip(best, best_indices)):
            decoded[(i), :sent_lengths[i]] = hypo
            if indices is not None:
                indices[(i), :len(best_idx)] = paddle.to_tensor(data=best_idx)
            if sent_lengths[i] < sent_max_len:
                decoded[i, sent_lengths[i]] = eos_token_id[0]
        return UserDict({'sequences': decoded, 'sequence_scores':
            best_scores, 'beam_indices': indices})


class ConstrainedBeamSearchScorer(BeamScorer):
    """
    [`BeamScorer`] implementing constrained beam search decoding.


    Args:
        batch_size (`int`):
            Batch Size of `input_ids` for which standard beam search decoding is run in parallel.
        num_beams (`int`):
            Number of beams for beam search.
        constraints (`List[Constraint]`):
            A list of positive constraints represented as `Constraint` objects that must be fulfilled in the generation
            output. For more information, the documentation of [`Constraint`] should be read.
        device (`torch.device`):
            Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be
            allocated.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        do_early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        num_beam_hyps_to_keep (`int`, *optional*, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            [`~transformer.BeamSearchScorer.finalize`].
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        max_length (`int`, *optional*):
            The maximum length of the sequence to be generated.
    """

    def __init__(self, batch_size: int, num_beams: int, constraints: List[
        Constraint], device: str, length_penalty: Optional[float] = 1.0,
                 do_early_stopping: Optional[Union[bool, str]] = False,
                 num_beam_hyps_to_keep: Optional[int] = 1, num_beam_groups: Optional[
                int] = 1, max_length: Optional[int] = None):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constraints = constraints
        self._is_init = False
        self._beam_hyps = [BeamHypotheses(num_beams=self.num_beams,
                                          length_penalty=self.length_penalty, early_stopping=self.
                                          do_early_stopping, max_length=max_length) for _ in range(
            batch_size)]
        self._done = paddle.to_tensor(data=[(False) for _ in range(
            batch_size)], dtype='bool', place=self.device)
        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f'`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead.'
            )
        if not isinstance(num_beam_groups, int
                          ) or num_beam_groups > num_beams or num_beams % num_beam_groups != 0:
            raise ValueError(
                f'`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}.'
            )

    @property
    def is_done(self) -> bool:
        return self._done.astype('bool').all()

    def make_constraint_states(self, n):
        return [ConstraintListState([constraint.copy() for constraint in
                                     self.constraints]) for _ in range(n)]

    def check_completes_constraints(self, sequence):
        new_state = self.make_constraint_states(1)[0]
        new_state.reset(sequence)
        return new_state.completed

    def process(self, input_ids: paddle.Tensor, next_scores: paddle.Tensor,
                next_tokens: paddle.Tensor, next_indices: paddle.Tensor,
                scores_for_all_vocab: paddle.Tensor, pad_token_id: Optional[int] =
                None, eos_token_id: Optional[Union[int, List[int]]] = None) -> Tuple[
        paddle.Tensor]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size * num_beams, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using any class inheriting from [`PreTrainedTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            next_scores (`torch.FloatTensor` of shape `(batch_size, 2 * num_beams)`):
                Current scores of the top `2 * num_beams` non-finished beam hypotheses.
            next_tokens (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                `input_ids` of the tokens corresponding to the top `2 * num_beams` non-finished beam hypotheses.
            next_indices (`torch.LongTensor` of shape `(batch_size, 2 * num_beams)`):
                Beam indices indicating to which beam hypothesis the `next_tokens` correspond.
            scores_for_all_vocab (`torch.FloatTensor` of shape `(batch_size * num_beams, sequence_length)`):
                The scores of all tokens in the vocabulary for each of the beam hypotheses.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        Return:
            `UserDict`: A dictionary composed of the fields as defined above:

                - **next_beam_scores** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Updated scores of
                  all
                non-finished beams.

                - **next_beam_tokens** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Next tokens to be
                  added
                to the non-finished beam_hypotheses.
                - **next_beam_indices** (`torch.FloatTensor` of shape `(batch_size * num_beams)`) -- Beam indices
                indicating to which beam the next tokens shall be added.
        """
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        if not batch_size == input_ids.shape[0] // self.group_size:
            if self.num_beam_groups > 1:
                raise ValueError(
                    f'A group beam size of {input_ids.shape[0]} is used as the input, but a group beam size of {self.group_size} is expected by the beam scorer.'
                )
            else:
                raise ValueError(
                    f'A beam size of {input_ids.shape[0]} is used as the input, but a beam size of {self.group_size} is expected by the beam scorer.'
                )
        device = input_ids.place
        next_beam_scores = paddle.zeros(shape=(batch_size, self.group_size),
                                        dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros(shape=(batch_size, self.group_size),
                                        dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros(shape=(batch_size, self.group_size
                                                ), dtype=next_indices.dtype)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                if self.num_beams < len(beam_hyp):
                    raise ValueError(
                        f'Batch can only be done if at least {self.num_beams} beams have been generated'
                    )
                if eos_token_id is None or pad_token_id is None:
                    raise ValueError(
                        'Generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                    )
                next_beam_scores[(batch_idx), :] = 0
                next_beam_tokens[(batch_idx), :] = pad_token_id
                next_beam_indices[(batch_idx), :] = 0
                continue
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index
                                  ) in enumerate(zip(next_tokens[batch_idx], next_scores[
                batch_idx], next_indices[batch_idx])):
                batch_beam_idx = batch_idx * self.group_size + next_index
                if eos_token_id is not None and next_token.item(
                ) in eos_token_id:
                    is_beam_token_worse_than_top_num_beams = (
                            beam_token_rank >= self.group_size)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    completes_constraint = self.check_completes_constraints(
                        input_ids[batch_beam_idx].cpu().tolist())
                    if completes_constraint:
                        beam_hyp.add(input_ids[batch_beam_idx].clone(),
                                     next_score.item())
                else:
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1
                if beam_idx == self.group_size:
                    break
            new_scores, new_tokens, new_indices = (self.
                                                   step_sentence_constraint(batch_idx, input_ids,
                                                                            scores_for_all_vocab,
                                                                            next_beam_scores[batch_idx],
                                                                            next_beam_tokens[batch_idx],
                                                                            next_beam_indices[batch_idx]))
            next_beam_scores[batch_idx] = new_scores
            next_beam_tokens[batch_idx] = new_tokens
            next_beam_indices[batch_idx] = new_indices
            if beam_idx < self.group_size:
                raise ValueError(
                    f'At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected.'
                )
            cur_len += 1
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len)
            return UserDict({'next_beam_scores': next_beam_scores.reshape([-1]),
                             'next_beam_tokens': next_beam_tokens.reshape([-1]),
                             'next_beam_indices': next_beam_indices.reshape([-1])})


def step_sentence_constraint(self, batch_idx: int, input_ids: paddle.
                             Tensor, vocab_scores: paddle.Tensor, sent_beam_scores: paddle.
                             Tensor, sent_beam_tokens: paddle.Tensor, sent_beam_indices: paddle.
                             Tensor, push_progress: bool = False):
    orig_len = sent_beam_indices.shape[0]
    device = sent_beam_indices.place
    topk_contraint_states = self.make_constraint_states(orig_len)
    advance_constraint_states = self.make_constraint_states(orig_len)
    sidx, eidx = batch_idx * orig_len, (batch_idx + 1) * orig_len
    this_batch_input_ids = input_ids[sidx:eidx]
    this_batch_token_scores = vocab_scores[sidx:eidx]
    full_hypotheses = paddle.concat(x=(input_ids[sent_beam_indices],
                                       sent_beam_tokens.unsqueeze(axis=-1)), axis=-1)
    track_new = {'new_seqs': full_hypotheses.tolist(), 'new_states': [],
                 'new_indices': [], 'new_tokens': [], 'new_scores': []}
    for seq_idx, pre_seq in enumerate(this_batch_input_ids):
        topk_state = topk_contraint_states[seq_idx]
        topk_state.reset(full_hypotheses[seq_idx].cpu().tolist())
        advance_state = advance_constraint_states[seq_idx]
        advance_state.reset(pre_seq.cpu().tolist())
        if not advance_state.completed:
            advance_tokens = paddle.to_tensor(data=advance_state.
                                              advance(), dtype='int64').to(device)
            for advance_token in advance_tokens:
                new_state = advance_state.copy(stateful=True)
                new_state.add(advance_token.cpu().tolist())
                advance_seq = paddle.concat(x=(pre_seq, advance_token.
                                               unsqueeze(axis=0)), axis=-1).cpu().tolist()
                if advance_seq not in track_new['new_seqs']:
                    track_new['new_seqs'].append(advance_seq)
                    track_new['new_indices'].append(sidx + seq_idx)
                    track_new['new_tokens'].append(advance_token)
                    track_new['new_scores'].append(this_batch_token_scores
                                                   [seq_idx].take(advance_token))
                    track_new['new_states'].append(new_state)
        elif push_progress:
            new_score, new_token = paddle.max(x=this_batch_token_scores
            [seq_idx], axis=0), paddle.argmax(x=
                                              this_batch_token_scores[seq_idx], axis=0)
            advance_seq = paddle.concat(x=(pre_seq, new_token.unsqueeze
            (axis=0)), axis=-1)
            advance_state = advance_constraint_states[seq_idx]
            advance_seq = advance_seq.cpu().tolist()
            advance_state.reset(advance_seq)
            if advance_seq not in track_new['new_seqs']:
                track_new['new_seqs'].append(advance_seq)
                track_new['new_indices'].append(seq_idx)
                track_new['new_tokens'].append(new_token)
                track_new['new_scores'].append(new_score)
                track_new['new_states'].append(advance_state)
    if len(track_new['new_indices']) > 0:
        new_indices = paddle.to_tensor(data=track_new['new_indices']).to(
            device)
        new_tokens = paddle.stack(x=track_new['new_tokens']).to(device)
        new_scores = paddle.stack(x=track_new['new_scores']).to(device)
        all_states = topk_contraint_states + track_new['new_states']
        all_tokens = paddle.concat(x=(sent_beam_tokens, new_tokens),
                                   axis=-1)
        all_scores = paddle.concat(x=(sent_beam_scores, new_scores),
                                   axis=-1)
        all_banks = paddle.to_tensor(data=[one.get_bank() for one in
                                           all_states]).to(device)
        zipped = all_banks * 100 + all_scores
        indices = (paddle.sort(descending=True, x=zipped), paddle.
                   argsort(descending=True, x=zipped)).indices
        sorted_banks = all_banks[indices]
        counter = -1
        cur_bank = sorted_banks[0]
        increments = []
        for bank in sorted_banks:
            if bank == cur_bank:
                counter += 1
            else:
                counter = 0
                cur_bank = bank
            increments.append(counter)
        rearrangers = paddle.to_tensor(data=np.argsort(increments, kind
        ='mergesort'))
        indices = indices[rearrangers][:orig_len]
        sent_beam_scores = all_scores[indices]
        sent_beam_tokens = all_tokens[indices]
        sent_beam_indices = paddle.concat(x=(sent_beam_indices,
                                             new_indices))[indices]
    return sent_beam_scores, sent_beam_tokens, sent_beam_indices


def finalize(self, input_ids: paddle.Tensor, final_beam_scores: paddle.
             Tensor, final_beam_tokens: paddle.Tensor, final_beam_indices:
paddle.Tensor, max_length: int, pad_token_id: Optional[int] = None,
             eos_token_id: Optional[Union[int, List[int]]] = None) -> Tuple[paddle.
Tensor]:
    batch_size = len(self._beam_hyps)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    for batch_idx, beam_hyp in enumerate(self._beam_hyps):
        if self._done[batch_idx]:
            continue
        ids_collect = []
        for beam_id in range(self.num_beams):
            batch_beam_idx = batch_idx * self.num_beams + beam_id
            final_score = final_beam_scores[batch_beam_idx].item()
            final_tokens = input_ids[batch_beam_idx]
            completes_constraint = self.check_completes_constraints(
                final_tokens.cpu().tolist())
            if completes_constraint:
                beam_hyp.add(final_tokens, final_score)
                ids_collect.append(beam_id)
        if len(ids_collect) < self.num_beam_hyps_to_keep:
            for beam_id in range(self.num_beams):
                if beam_id not in ids_collect:
                    batch_beam_idx = batch_idx * self.num_beams + beam_id
                    final_score = final_beam_scores[batch_beam_idx].item()
                    final_tokens = input_ids[batch_beam_idx]
                    beam_hyp.add(final_tokens, final_score)
                if len(ids_collect) >= self.num_beam_hyps_to_keep:
                    break
    sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
    best = []
    best_scores = paddle.zeros(shape=batch_size * self.
                               num_beam_hyps_to_keep, dtype='float32')
    for i, beam_hyp in enumerate(self._beam_hyps):
        sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
        for j in range(self.num_beam_hyps_to_keep):
            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp
                                                                   )
            best.append(best_hyp)
            best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
    sent_lengths_max = sent_lengths.max().item() + 1
    sent_max_len = min(sent_lengths_max, max_length
                       ) if max_length is not None else sent_lengths_max
    decoded: paddle.Tensor = input_ids.new(batch_size * self.
                                           num_beam_hyps_to_keep, sent_max_len)
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, '`pad_token_id` has to be defined'
        decoded.fill_(value=pad_token_id)
    for i, hypo in enumerate(best):
        decoded[(i), :sent_lengths[i]] = hypo
        if sent_lengths[i] < sent_max_len:
            decoded[i, sent_lengths[i]] = eos_token_id[0]
    return UserDict({'sequences': decoded, 'sequence_scores': best_scores})


class BeamHypotheses:

    def __init__(self, num_beams: int, length_penalty: float,
                 early_stopping: bool, max_length: Optional[int] = None):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0
        if not isinstance(self.early_stopping, bool
                          ) and self.max_length is None:
            raise ValueError(
                'When `do_early_stopping` is set to a string, `max_length` must be defined. Ensure it is passed to the BeamScorer class instance at initialization time.'
            )

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: paddle.Tensor, sum_logprobs: float, beam_indices:
    Optional[paddle.Tensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / hyp.shape[-1] ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in
                                             enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        if self.early_stopping is True:
            return True
        elif self.early_stopping is False:
            highest_attainable_score = (best_sum_logprobs / cur_len ** self
                                        .length_penalty)
            ret = self.worst_score >= highest_attainable_score
            return ret
        else:
            if self.length_penalty > 0.0:
                highest_attainable_score = (best_sum_logprobs / self.
                                            max_length ** self.length_penalty)
            else:
                highest_attainable_score = (best_sum_logprobs / cur_len **
                                            self.length_penalty)
            ret = self.worst_score >= highest_attainable_score
            return ret
