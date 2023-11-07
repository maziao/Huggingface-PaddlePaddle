# import sys
# sys.path.append(
#     '/Users/maziao/Documents/DataHammerGroup/site-packages/transformers-paddle/generation/utils'
#     )
# import paddle_aux
# import paddle
# import copy
# import inspect
# import warnings
# from dataclasses import dataclass
# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from ..deepspeed import is_deepspeed_zero3_enabled
# from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
# from ..models.auto import MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING, MODEL_FOR_VISION_2_SEQ_MAPPING
# from ..utils import ModelOutput, logging
# from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
# from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
# from .configuration_utils import GenerationConfig
# from .logits_process import EncoderNoRepeatNGramLogitsProcessor, EncoderRepetitionPenaltyLogitsProcessor, EpsilonLogitsWarper, EtaLogitsWarper, ExponentialDecayLengthPenalty, ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor, ForceTokensLogitsProcessor, HammingDiversityLogitsProcessor, InfNanRemoveLogitsProcessor, LogitNormalization, LogitsProcessorList, MinLengthLogitsProcessor, MinNewTokensLengthLogitsProcessor, NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor, PrefixConstrainedLogitsProcessor, RepetitionPenaltyLogitsProcessor, SuppressTokensAtBeginLogitsProcessor, SuppressTokensLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, TypicalLogitsWarper
# from .stopping_criteria import MaxLengthCriteria, MaxTimeCriteria, StoppingCriteria, StoppingCriteriaList, validate_stopping_criteria
# if TYPE_CHECKING:
#     from .streamers import BaseStreamer
# logger = logging.get_logger(__name__)
#
#
# @dataclass
# class GreedySearchDecoderOnlyOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using greedy search.
#
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class ContrastiveSearchEncoderDecoderOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using contrastive search.
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
#             sequence_length, sequence_length)`.
#         encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`.
#         decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     encoder_attentions: Optional[Tuple[paddle.Tensor]] = None
#     encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
#     decoder_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     cross_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     decoder_hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class ContrastiveSearchDecoderOnlyOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using contrastive search.
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when
#         `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is
#         passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class GreedySearchEncoderDecoderOutput(ModelOutput):
#     """
#     Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
#     weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
#     encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
#
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
#         encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
#             sequence_length, sequence_length)`.
#         encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`.
#         decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     encoder_attentions: Optional[Tuple[paddle.Tensor]] = None
#     encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
#     decoder_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     cross_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     decoder_hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class SampleDecoderOnlyOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using sampling.
#
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(num_return_sequences*batch_size, num_heads, generated_length,
#             sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(num_return_sequences*batch_size, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class SampleEncoderDecoderOutput(ModelOutput):
#     """
#     Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
#     the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
#     attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
#
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
#             at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
#             each generated token), with each tensor of shape `(batch_size*num_return_sequences, config.vocab_size)`.
#         encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape
#             `(batch_size*num_return_sequences, num_heads, sequence_length, sequence_length)`.
#         encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size*num_return_sequences, sequence_length, hidden_size)`.
#         decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_return_sequences, num_heads, generated_length,
#             sequence_length)`.
#         cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_return_sequences, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     encoder_attentions: Optional[Tuple[paddle.Tensor]] = None
#     encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
#     decoder_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     cross_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     decoder_hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class BeamSearchDecoderOnlyOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using beam search.
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Final beam scores of the generated `sequences`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
#             of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
#             Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
#             with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
#         beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
#             `(batch_size*num_return_sequences, sequence_length)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     sequences_scores: Optional[paddle.Tensor] = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     beam_indices: Optional[paddle.Tensor] = None
#     attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class BeamSearchEncoderDecoderOutput(ModelOutput):
#     """
#     Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
#     of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
#     attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Final beam scores of the generated `sequences`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
#             of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
#             Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
#             with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
#         beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
#             `(batch_size*num_return_sequences, sequence_length)`.
#         encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
#             sequence_length, sequence_length)`.
#         encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
#         decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
#             sequence_length)`.
#         cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     sequences_scores: Optional[paddle.Tensor] = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     beam_indices: Optional[paddle.Tensor] = None
#     encoder_attentions: Optional[Tuple[paddle.Tensor]] = None
#     encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
#     decoder_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     cross_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     decoder_hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class BeamSampleDecoderOnlyOutput(ModelOutput):
#     """
#     Base class for outputs of decoder-only generation models using beam sample.
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         sequences_scores (`torch.FloatTensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Final beam scores of the generated `sequences`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
#             of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
#             Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
#             with each tensor of shape `(batch_size*num_beams*num_return_sequences, config.vocab_size)`.
#         beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
#             `(batch_size*num_return_sequences, sequence_length)`.
#         attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
#         hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     sequences_scores: Optional[paddle.Tensor] = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     beam_indices: Optional[paddle.Tensor] = None
#     attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# @dataclass
# class BeamSampleEncoderDecoderOutput(ModelOutput):
#     """
#     Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention
#     weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
#     encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)
#
#     Args:
#         sequences (`torch.LongTensor` of shape `(batch_size*num_beams, sequence_length)`):
#             The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
#             if all batches finished early due to the `eos_token_id`.
#         sequences_scores (`torch.FloatTensor` of shape `(batch_size * num_return_sequence)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Final beam scores of the generated `sequences`.
#         scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
#             of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
#             Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
#             with each tensor of shape `(batch_size*num_beams, config.vocab_size)`).
#         beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
#             Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
#             `(batch_size*num_return_sequences, sequence_length)`.
#         encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
#             sequence_length, sequence_length)`.
#         encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size*num_beams, sequence_length, hidden_size)`.
#         decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
#         cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
#         decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
#             `torch.FloatTensor` of shape `(batch_size*num_beams, generated_length, hidden_size)`.
#     """
#     sequences: paddle.Tensor = None
#     sequences_scores: Optional[paddle.Tensor] = None
#     scores: Optional[Tuple[paddle.Tensor]] = None
#     beam_indices: Optional[paddle.Tensor] = None
#     encoder_attentions: Optional[Tuple[paddle.Tensor]] = None
#     encoder_hidden_states: Optional[Tuple[paddle.Tensor]] = None
#     decoder_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     cross_attentions: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#     decoder_hidden_states: Optional[Tuple[Tuple[paddle.Tensor]]] = None
#
#
# GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput,
#     GreedySearchDecoderOnlyOutput]
# SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
# BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput,
#     BeamSearchDecoderOnlyOutput]
# BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput,
#     BeamSampleDecoderOnlyOutput]
# ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput,
#     ContrastiveSearchDecoderOnlyOutput]
# GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput,
#     BeamSampleOutput, ContrastiveSearchOutput]
#
#
# class GenerationMixin:
#     """
#     A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].
#
#     The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
#         - *greedy decoding* by calling [`~generation.GenerationMixin.greedy_search`] if `num_beams=1` and
#           `do_sample=False`
#         - *contrastive search* by calling [`~generation.GenerationMixin.contrastive_search`] if `penalty_alpha>0` and
#           `top_k>1`
#         - *multinomial sampling* by calling [`~generation.GenerationMixin.sample`] if `num_beams=1` and
#           `do_sample=True`
#         - *beam-search decoding* by calling [`~generation.GenerationMixin.beam_search`] if `num_beams>1` and
#           `do_sample=False`
#         - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin.beam_sample`] if `num_beams>1`
#           and `do_sample=True`
#         - *diverse beam-search decoding* by calling [`~generation.GenerationMixin.group_beam_search`], if `num_beams>1`
#           and `num_beam_groups>1`
#         - *constrained beam-search decoding* by calling [`~generation.GenerationMixin.constrained_beam_search`], if
#           `constraints!=None` or `force_words_ids!=None`
#
#     You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
#     learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
#     """
#
#     def prepare_inputs_for_generation(self, *args, **kwargs):
#         raise NotImplementedError(
#             'A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`.'
#             )
#
#     def _prepare_model_inputs(self, inputs: Optional[paddle.Tensor]=None,
#         bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str,
#         paddle.Tensor]]=None) ->Tuple[paddle.Tensor, Optional[str], Dict[
#         str, paddle.Tensor]]:
#         """
#         This function extracts the model-specific `inputs` for generation.
#         """
#         if self.config.is_encoder_decoder and hasattr(self, 'encoder'
#             ) and self.encoder.main_input_name != self.main_input_name:
#             input_name = self.encoder.main_input_name
#         else:
#             input_name = self.main_input_name
#         model_kwargs = {k: v for k, v in model_kwargs.items() if v is not
#             None or k != input_name}
#         inputs_kwarg = model_kwargs.pop(input_name, None)
#         if inputs_kwarg is not None and inputs is not None:
#             raise ValueError(
#                 f'`inputs`: {inputs}` were passed alongside {input_name} which is not allowed.Make sure to either pass {inputs} or {input_name}=...'
#                 )
#         elif inputs_kwarg is not None:
#             inputs = inputs_kwarg
#         if input_name == 'input_ids' and 'inputs_embeds' in model_kwargs:
#             if not self.config.is_encoder_decoder:
#                 has_inputs_embeds_forwarding = 'inputs_embeds' in set(inspect
#                     .signature(self.prepare_inputs_for_generation).
#                     parameters.keys())
#                 if not has_inputs_embeds_forwarding:
#                     raise ValueError(
#                         f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} doesn't have its forwarding implemented. See the GPT2 implementation for an example (https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
#                         )
#                 model_kwargs['input_ids'
#                     ] = self._maybe_initialize_input_ids_for_generation(inputs,
#                     bos_token_id, model_kwargs=model_kwargs)
#             elif inputs is not None:
#                 raise ValueError(
#                     'You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.'
#                     )
#             inputs, input_name = model_kwargs['inputs_embeds'], 'inputs_embeds'
#         inputs = self._maybe_initialize_input_ids_for_generation(inputs,
#             bos_token_id, model_kwargs)
#         return inputs, input_name, model_kwargs
#
#     def adjust_logits_during_generation(self, logits: paddle.Tensor, **kwargs
#         ) ->paddle.Tensor:
#         """
#         Implement in subclasses of [`PreTrainedModel`] for custom behavior to adjust the logits in the generate method.
#         """
#         return logits
#
#     def _maybe_initialize_input_ids_for_generation(self, inputs: Optional[
#         paddle.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs:
#         Optional[Dict[str, paddle.Tensor]]=None) ->paddle.Tensor:
#         """Initializes input ids for generation, if necessary."""
#         if inputs is not None:
#             return inputs
#         encoder_outputs = model_kwargs.get('encoder_outputs')
#         if self.config.is_encoder_decoder and encoder_outputs is not None:
#             shape = encoder_outputs.last_hidden_state.size()[:-1]
#             return paddle.ones(shape=shape, dtype='int64') * -100
#         if bos_token_id is None:
#             raise ValueError(
#                 '`bos_token_id` has to be defined when no `input_ids` are provided.'
#                 )
#         batch_size = 1
#         for value in model_kwargs.values():
#             if isinstance(value, paddle.Tensor):
#                 batch_size = value.shape[0]
#                 break
#         return paddle.ones(shape=(batch_size, 1), dtype='int64') * bos_token_id
#
#     def _prepare_attention_mask_for_generation(self, inputs: paddle.Tensor,
#         pad_token_id: Optional[int], eos_token_id: Optional[Union[int, List
#         [int]]]) ->paddle.Tensor:
#         is_input_ids = len(inputs.shape) == 2 and inputs.dtype in ['int32',
#             'int64']
#         is_pad_token_in_inputs = (pad_token_id is not None and pad_token_id in
#             inputs)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None or
#             pad_token_id not in eos_token_id)
#         if (is_input_ids and is_pad_token_in_inputs and
#             is_pad_token_not_equal_to_eos_token_id):
#             return inputs.not_equal(y=paddle.to_tensor(pad_token_id)).astype(
#                 dtype='int64')
#         else:
#             return paddle.ones(shape=inputs.shape[:2], dtype='int64')
#
#     def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor:
#         paddle.Tensor, model_kwargs, model_input_name: Optional[str]=None
#         ) ->Dict[str, Any]:
#         encoder = self.get_encoder()
#         irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
#         encoder_kwargs = {argument: value for argument, value in
#             model_kwargs.items() if not any(argument.startswith(p) for p in
#             irrelevant_prefix)}
#         encoder_signature = set(inspect.signature(encoder.forward).parameters)
#         encoder_accepts_wildcard = ('kwargs' in encoder_signature or
#             'model_kwargs' in encoder_signature)
#         if not encoder_accepts_wildcard:
#             encoder_kwargs = {argument: value for argument, value in
#                 encoder_kwargs.items() if argument in encoder_signature}
#         model_input_name = (model_input_name if model_input_name is not
#             None else self.main_input_name)
#         encoder_kwargs['return_dict'] = True
#         encoder_kwargs[model_input_name] = inputs_tensor
#         model_kwargs['encoder_outputs']: ModelOutput = encoder(**encoder_kwargs
#             )
#         return model_kwargs
#
#     def _prepare_decoder_input_ids_for_generation(self, batch_size: int,
#         decoder_start_token_id: int=None, bos_token_id: int=None,
#         model_kwargs: Optional[Dict[str, paddle.Tensor]]=None, device: str=None
#         ) ->paddle.Tensor:
#         if model_kwargs is not None and 'decoder_input_ids' in model_kwargs:
#             return model_kwargs.pop('decoder_input_ids')
#         else:
#             decoder_start_token_id = self._get_decoder_start_token_id(
#                 decoder_start_token_id, bos_token_id)
#             if device is None:
#                 device = self.device
#             return paddle.ones(shape=(batch_size, 1), dtype='int64'
#                 ) * decoder_start_token_id
#
#     def _get_decoder_start_token_id(self, decoder_start_token_id: int=None,
#         bos_token_id: int=None) ->int:
#         decoder_start_token_id = (decoder_start_token_id if
#             decoder_start_token_id is not None else self.generation_config.
#             decoder_start_token_id)
#         bos_token_id = (bos_token_id if bos_token_id is not None else self.
#             generation_config.bos_token_id)
#         if decoder_start_token_id is not None:
#             return decoder_start_token_id
#         elif bos_token_id is not None:
#             return bos_token_id
#         raise ValueError(
#             '`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.'
#             )
#
#     @staticmethod
#     def _expand_inputs_for_generation(expand_size: int=1,
#         is_encoder_decoder: bool=False, input_ids: Optional[paddle.Tensor]=
#         None, **model_kwargs) ->Tuple[paddle.Tensor, Dict[str, Any]]:
#         """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
#
#         def _expand_dict_for_generation(dict_to_expand):
#             for key in dict_to_expand:
#                 if dict_to_expand[key] is not None and isinstance(
#                     dict_to_expand[key], paddle.Tensor):
#                     dict_to_expand[key] = dict_to_expand[key
#                         ].repeat_interleave(repeats=expand_size, axis=0)
#             return dict_to_expand
#         if input_ids is not None:
#             input_ids = input_ids.repeat_interleave(repeats=expand_size, axis=0
#                 )
#         model_kwargs = _expand_dict_for_generation(model_kwargs)
#         if is_encoder_decoder:
#             if model_kwargs.get('encoder_outputs') is None:
#                 raise ValueError(
#                     'If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.'
#                     )
#             model_kwargs['encoder_outputs'] = _expand_dict_for_generation(
#                 model_kwargs['encoder_outputs'])
#         return input_ids, model_kwargs
#
#     def _extract_past_from_model_output(self, outputs: ModelOutput,
#         standardize_cache_format: bool=False):
#         past_key_values = None
#         if 'past_key_values' in outputs:
#             past_key_values = outputs.past_key_values
#         elif 'mems' in outputs:
#             past_key_values = outputs.mems
#         elif 'past_buckets_states' in outputs:
#             past_key_values = outputs.past_buckets_states
#         if standardize_cache_format and hasattr(self,
#             '_convert_to_standard_cache'):
#             batch_size = outputs.logits.shape[0]
#             past_key_values = self._convert_to_standard_cache(past_key_values,
#                 batch_size=batch_size)
#         return past_key_values
#
#     def _update_model_kwargs_for_generation(self, outputs: ModelOutput,
#         model_kwargs: Dict[str, Any], is_encoder_decoder: bool=False,
#         standardize_cache_format: bool=False) ->Dict[str, Any]:
#         model_kwargs['past_key_values'] = self._extract_past_from_model_output(
#             outputs, standardize_cache_format=standardize_cache_format)
#         if 'token_type_ids' in model_kwargs:
#             token_type_ids = model_kwargs['token_type_ids']
#             model_kwargs['token_type_ids'] = paddle.concat(x=[
#                 token_type_ids, token_type_ids[:, (-1)].unsqueeze(axis=-1)],
#                 axis=-1)
#         if not is_encoder_decoder:
#             if 'attention_mask' in model_kwargs:
#                 attention_mask = model_kwargs['attention_mask']
#                 model_kwargs['attention_mask'] = paddle.concat(x=[
#                     attention_mask, paddle.ones(shape=(attention_mask.shape
#                     [0], 1), dtype=attention_mask.dtype)], axis=-1)
#         elif 'decoder_attention_mask' in model_kwargs:
#             decoder_attention_mask = model_kwargs['decoder_attention_mask']
#             model_kwargs['decoder_attention_mask'] = paddle.concat(x=[
#                 decoder_attention_mask, paddle.ones(shape=(
#                 decoder_attention_mask.shape[0], 1), dtype=
#                 decoder_attention_mask.dtype)], axis=-1)
#         return model_kwargs
#
#     def _reorder_cache(self, past_key_values, beam_idx):
#         raise NotImplementedError(
#             f'Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to enable beam search for {self.__class__}'
#             )
#
#     def _get_logits_warper(self, generation_config: GenerationConfig
#         ) ->LogitsProcessorList:
#         """
#         This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
#         used for multinomial sampling.
#         """
#         warpers = LogitsProcessorList()
#         if (generation_config.temperature is not None and generation_config
#             .temperature != 1.0):
#             warpers.append(TemperatureLogitsWarper(generation_config.
#                 temperature))
#         min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1
#         if (generation_config.top_k is not None and generation_config.top_k !=
#             0):
#             warpers.append(TopKLogitsWarper(top_k=generation_config.top_k,
#                 min_tokens_to_keep=min_tokens_to_keep))
#         if (generation_config.top_p is not None and generation_config.top_p <
#             1.0):
#             warpers.append(TopPLogitsWarper(top_p=generation_config.top_p,
#                 min_tokens_to_keep=min_tokens_to_keep))
#         if (generation_config.typical_p is not None and generation_config.
#             typical_p < 1.0):
#             warpers.append(TypicalLogitsWarper(mass=generation_config.
#                 typical_p, min_tokens_to_keep=min_tokens_to_keep))
#         if (generation_config.epsilon_cutoff is not None and 0.0 <
#             generation_config.epsilon_cutoff < 1.0):
#             warpers.append(EpsilonLogitsWarper(epsilon=generation_config.
#                 epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep))
#         if (generation_config.eta_cutoff is not None and 0.0 <
#             generation_config.eta_cutoff < 1.0):
#             warpers.append(EtaLogitsWarper(epsilon=generation_config.
#                 eta_cutoff, min_tokens_to_keep=min_tokens_to_keep))
#         if generation_config.renormalize_logits is True:
#             warpers.append(LogitNormalization())
#         return warpers
#
#     def _get_logits_processor(self, generation_config: GenerationConfig,
#         input_ids_seq_length: int, encoder_input_ids: paddle.Tensor,
#         prefix_allowed_tokens_fn: Callable[[int, paddle.Tensor], List[int]],
#         logits_processor: Optional[LogitsProcessorList]) ->LogitsProcessorList:
#         """
#         This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
#         instances used to modify the scores of the language model head.
#         """
#         processors = LogitsProcessorList()
#         if (generation_config.diversity_penalty is not None and
#             generation_config.diversity_penalty > 0.0):
#             processors.append(HammingDiversityLogitsProcessor(
#                 diversity_penalty=generation_config.diversity_penalty,
#                 num_beams=generation_config.num_beams, num_beam_groups=
#                 generation_config.num_beam_groups))
#         if (generation_config.encoder_repetition_penalty is not None and
#             generation_config.encoder_repetition_penalty != 1.0):
#             processors.append(EncoderRepetitionPenaltyLogitsProcessor(
#                 penalty=generation_config.encoder_repetition_penalty,
#                 encoder_input_ids=encoder_input_ids))
#         if (generation_config.repetition_penalty is not None and
#             generation_config.repetition_penalty != 1.0):
#             processors.append(RepetitionPenaltyLogitsProcessor(penalty=
#                 generation_config.repetition_penalty))
#         if (generation_config.no_repeat_ngram_size is not None and
#             generation_config.no_repeat_ngram_size > 0):
#             processors.append(NoRepeatNGramLogitsProcessor(
#                 generation_config.no_repeat_ngram_size))
#         if (generation_config.encoder_no_repeat_ngram_size is not None and
#             generation_config.encoder_no_repeat_ngram_size > 0):
#             if self.config.is_encoder_decoder:
#                 processors.append(EncoderNoRepeatNGramLogitsProcessor(
#                     generation_config.encoder_no_repeat_ngram_size,
#                     encoder_input_ids))
#             else:
#                 raise ValueError(
#                     "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
#                     )
#         if generation_config.bad_words_ids is not None:
#             processors.append(NoBadWordsLogitsProcessor(generation_config.
#                 bad_words_ids, generation_config.eos_token_id))
#         if (generation_config.min_length is not None and generation_config.
#             eos_token_id is not None and generation_config.min_length > 0):
#             processors.append(MinLengthLogitsProcessor(generation_config.
#                 min_length, generation_config.eos_token_id))
#         if (generation_config.min_new_tokens is not None and
#             generation_config.eos_token_id is not None and
#             generation_config.min_new_tokens > 0):
#             processors.append(MinNewTokensLengthLogitsProcessor(
#                 input_ids_seq_length, generation_config.min_new_tokens,
#                 generation_config.eos_token_id))
#         if prefix_allowed_tokens_fn is not None:
#             processors.append(PrefixConstrainedLogitsProcessor(
#                 prefix_allowed_tokens_fn, generation_config.num_beams //
#                 generation_config.num_beam_groups))
#         if generation_config.forced_bos_token_id is not None:
#             processors.append(ForcedBOSTokenLogitsProcessor(
#                 generation_config.forced_bos_token_id))
#         if generation_config.forced_eos_token_id is not None:
#             processors.append(ForcedEOSTokenLogitsProcessor(
#                 generation_config.max_length, generation_config.
#                 forced_eos_token_id))
#         if generation_config.remove_invalid_values is True:
#             processors.append(InfNanRemoveLogitsProcessor())
#         if generation_config.exponential_decay_length_penalty is not None:
#             processors.append(ExponentialDecayLengthPenalty(
#                 generation_config.exponential_decay_length_penalty,
#                 generation_config.eos_token_id, input_ids_seq_length))
#         if generation_config.suppress_tokens is not None:
#             processors.append(SuppressTokensLogitsProcessor(
#                 generation_config.suppress_tokens))
#         if generation_config.begin_suppress_tokens is not None:
#             begin_index = input_ids_seq_length
#             begin_index = (begin_index if input_ids_seq_length > 1 or
#                 generation_config.forced_bos_token_id is None else
#                 begin_index + 1)
#             if generation_config.forced_decoder_ids is not None:
#                 begin_index += generation_config.forced_decoder_ids[-1][0]
#             processors.append(SuppressTokensAtBeginLogitsProcessor(
#                 generation_config.begin_suppress_tokens, begin_index))
#         if generation_config.forced_decoder_ids is not None:
#             processors.append(ForceTokensLogitsProcessor(generation_config.
#                 forced_decoder_ids))
#         processors = self._merge_criteria_processor_list(processors,
#             logits_processor)
#         if generation_config.renormalize_logits is True:
#             processors.append(LogitNormalization())
#         return processors
#
#     def _get_stopping_criteria(self, generation_config: GenerationConfig,
#         stopping_criteria: Optional[StoppingCriteriaList]
#         ) ->StoppingCriteriaList:
#         criteria = StoppingCriteriaList()
#         if generation_config.max_length is not None:
#             criteria.append(MaxLengthCriteria(max_length=generation_config.
#                 max_length))
#         if generation_config.max_time is not None:
#             criteria.append(MaxTimeCriteria(max_time=generation_config.
#                 max_time))
#         criteria = self._merge_criteria_processor_list(criteria,
#             stopping_criteria)
#         return criteria
#
#     def _merge_criteria_processor_list(self, default_list: Union[
#         LogitsProcessorList, StoppingCriteriaList], custom_list: Union[
#         LogitsProcessorList, StoppingCriteriaList]) ->Union[
#         LogitsProcessorList, StoppingCriteriaList]:
#         if len(custom_list) == 0:
#             return default_list
#         for default in default_list:
#             for custom in custom_list:
#                 if type(custom) is type(default):
#                     object_type = 'stopping criteria' if isinstance(custom,
#                         StoppingCriteria) else 'logits processor'
#                     raise ValueError(
#                         f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to `generate`, but it has already been created with the values {default}. {default} has been created by passing the corresponding arguments to generate or by the model's config default values. If you just want to change the default values of {object_type} consider passing them as arguments to `generate` instead of using a custom {object_type}."
#                         )
#         default_list.extend(custom_list)
#         return default_list
#
#     def compute_transition_scores(self, sequences: paddle.Tensor, scores:
#         Tuple[paddle.Tensor], beam_indices: Optional[paddle.Tensor]=None,
#         normalize_logits: bool=False) ->paddle.Tensor:
#         """
#         Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
#         used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.
#
#         Parameters:
#             sequences (`torch.LongTensor`):
#                 The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
#                 shorter if all batches finished early due to the `eos_token_id`.
#             scores (`tuple(torch.FloatTensor)`):
#                 Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
#                 of log probabilities of tokens conditioned on log softmax of previously generated tokens Tuple of
#                 `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with
#                 each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
#             beam_indices (`torch.LongTensor`, *optional*):
#                 Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
#                 `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
#                 generate-time.
#             normalize_logits (`bool`, *optional*, defaults to `False`):
#                 Whether to normalize the logits (which, for legacy reasons, may be unnormalized).
#
#         Return:
#             `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
#                 the transition scores (logits)
#
#         Examples:
#
#         ```python
#         >>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
#         >>> import numpy as np
#
#         >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
#         >>> tokenizer.pad_token_id = tokenizer.eos_token_id
#         >>> inputs = tokenizer(["Today is"], return_tensors="pt")
#
#         >>> # Example 1: Print the scores for each token generated with Greedy Search
#         >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
#         >>> transition_scores = model.compute_transition_scores(
#         ...     outputs.sequences, outputs.scores, normalize_logits=True
#         ... )
#         >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
#         >>> # encoder-decoder models, like BART or T5.
#         >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
#         >>> generated_tokens = outputs.sequences[:, input_length:]
#         >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
#         ...     # | token | token string | logits | probability
#         ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
#         |   262 |  the     | -1.414 | 24.33%
#         |  1110 |  day     | -2.609 | 7.36%
#         |   618 |  when    | -2.010 | 13.40%
#         |   356 |  we      | -1.859 | 15.58%
#         |   460 |  can     | -2.508 | 8.14%
#
#         >>> # Example 2: Reconstruct the sequence scores from Beam Search
#         >>> outputs = model.generate(
#         ...     **inputs,
#         ...     max_new_tokens=5,
#         ...     num_beams=4,
#         ...     num_return_sequences=4,
#         ...     return_dict_in_generate=True,
#         ...     output_scores=True,
#         ... )
#         >>> transition_scores = model.compute_transition_scores(
#         ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
#         ... )
#         >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
#         >>> # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
#         >>> # use case, you might want to recompute it with `normalize_logits=True`.
#         >>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
#         >>> length_penalty = model.generation_config.length_penalty
#         >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
#         >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
#         True
#         ```"""
#         if beam_indices is None:
#             """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            beam_indices = paddle.arange(end=scores[0].shape[0]).view(-1, 1
#                 ).to(sequences.place)
#             beam_indices = beam_indices.expand(shape=[-1, len(scores)])
#         x = paddle.stack(x=scores).reshape(len(scores), -1)
#         perm_0 = list(range(x.ndim))
#         perm_0[0] = 1
#         perm_0[1] = 0
#         scores = x.transpose(perm=perm_0)
#         if normalize_logits:
#             scores = scores.reshape(-1, self.config.vocab_size, scores.
#                 shape[-1])
#             scores = paddle.nn.functional.log_softmax(x=scores, axis=1)
#             scores = scores.reshape(-1, scores.shape[-1])
#         beam_indices_mask = beam_indices < 0
#         max_beam_length = (1 - beam_indices_mask.astype(dtype='int64')).sum(
#             axis=-1).max()
#         beam_indices = beam_indices.clone()[:, :max_beam_length]
#         beam_indices_mask = beam_indices_mask[:, :max_beam_length]
#         beam_indices[beam_indices_mask] = 0
#         beam_sequence_indices = beam_indices * self.config.vocab_size
#         cut_idx = sequences.shape[-1] - max_beam_length
#         indices = sequences[:, cut_idx:] + beam_sequence_indices
#         transition_scores = scores.take_along_axis(axis=0, indices=indices)
#         transition_scores[beam_indices_mask] = 0
#         return transition_scores
#
#     def _validate_model_class(self):
#         """
#         Confirms that the model class is compatible with generation. If not, raises an exception that points to the
#         right class to use.
#         """
#         if not self.can_generate():
#             generate_compatible_mappings = [MODEL_FOR_CAUSAL_LM_MAPPING,
#                 MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
#                 MODEL_FOR_VISION_2_SEQ_MAPPING,
#                 MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
#                 MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING]
#             generate_compatible_classes = set()
#             for model_mapping in generate_compatible_mappings:
#                 supported_models = model_mapping.get(type(self.config),
#                     default=None)
#                 if supported_models is not None:
#                     generate_compatible_classes.add(supported_models.__name__)
#             exception_message = (
#                 f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as it doesn't have a language model head."
#                 )
#             if generate_compatible_classes:
#                 exception_message += (
#                     f' Please use one of the following classes instead: {generate_compatible_classes}'
#                     )
#             raise TypeError(exception_message)
#
#     def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
#         """Validates model kwargs for generation. Generate argument typos will also be caught here."""
#         if self.config.is_encoder_decoder:
#             for key in ['decoder_input_ids']:
#                 model_kwargs.pop(key, None)
#         unused_model_args = []
#         model_args = set(inspect.signature(self.
#             prepare_inputs_for_generation).parameters)
#         if 'kwargs' in model_args or 'model_kwargs' in model_args:
#             model_args |= set(inspect.signature(self.forward).parameters)
#         for key, value in model_kwargs.items():
#             if value is not None and key not in model_args:
#                 unused_model_args.append(key)
#         if unused_model_args:
#             raise ValueError(
#                 f'The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the generate arguments will also show up in this list)'
#                 )
#
#     @paddle.no_grad()
#     def generate(self, inputs: Optional[paddle.Tensor]=None,
#         generation_config: Optional[GenerationConfig]=None,
#         logits_processor: Optional[LogitsProcessorList]=None,
#         stopping_criteria: Optional[StoppingCriteriaList]=None,
#         prefix_allowed_tokens_fn: Optional[Callable[[int, paddle.Tensor],
#         List[int]]]=None, synced_gpus: Optional[bool]=None, streamer:
#         Optional['BaseStreamer']=None, **kwargs) ->Union[GenerateOutput,
#         paddle.Tensor]:
#         """
#
#         Generates sequences of token ids for models with a language modeling head.
#
#         <Tip warning={true}>
#
#         Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
#         model's default generation configuration. You can override any `generation_config` by passing the corresponding
#         parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.
#
#         For an overview of generation strategies and code examples, check out the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
#                 The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
#                 method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
#                 should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
#                 `input_ids`, `input_values`, `input_features`, or `pixel_values`.
#             generation_config (`~generation.GenerationConfig`, *optional*):
#                 The generation configuration to be used as base parametrization for the generation call. `**kwargs`
#                 passed to generate matching the attributes of `generation_config` will override them. If
#                 `generation_config` is not provided, the default will be used, which had the following loading
#                 priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
#                 configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
#                 default values, whose documentation should be checked to parameterize generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 Custom logits processors that complement the default logits processors built from arguments and
#                 generation config. If a logit processor is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
#                 generation config. If a stopping criteria is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
#                 If provided, this function constraints the beam search to allowed tokens only at each step. If not
#                 provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
#                 `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
#                 on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
#                 for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
#                 Retrieval](https://arxiv.org/abs/2010.00904).
#             synced_gpus (`bool`, *optional*):
#                 Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
#                 `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
#                 generating before other GPUs. Otherwise it'll be set to `False`.
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#
#             kwargs:
#                 Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
#                 forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
#                 specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
#
#         Return:
#             [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
#             or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
#
#                 If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GreedySearchDecoderOnlyOutput`],
#                     - [`~generation.SampleDecoderOnlyOutput`],
#                     - [`~generation.BeamSearchDecoderOnlyOutput`],
#                     - [`~generation.BeamSampleDecoderOnlyOutput`]
#
#                 If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GreedySearchEncoderDecoderOutput`],
#                     - [`~generation.SampleEncoderDecoderOutput`],
#                     - [`~generation.BeamSearchEncoderDecoderOutput`],
#                     - [`~generation.BeamSampleEncoderDecoderOutput`]
#         """
#         if synced_gpus is None:
#             if is_deepspeed_zero3_enabled(
# >>>>>>                ) and torch.distributed.get_world_size() > 1:
#                 synced_gpus = True
#             else:
#                 synced_gpus = False
#         self._validate_model_class()
#         if generation_config is None:
#             if self.generation_config._from_model_config:
#                 new_generation_config = GenerationConfig.from_model_config(self
#                     .config)
#                 if new_generation_config != self.generation_config:
#                     warnings.warn(
#                         'You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation)'
#                         )
#                     self.generation_config = new_generation_config
#             generation_config = self.generation_config
#         generation_config = copy.deepcopy(generation_config)
#         model_kwargs = generation_config.update(**kwargs)
#         generation_config.validate()
#         self._validate_model_kwargs(model_kwargs.copy())
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if (generation_config.pad_token_id is None and generation_config.
#             eos_token_id is not None):
#             if model_kwargs.get('attention_mask', None) is None:
#                 logger.warning(
#                     "The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
#                     )
#             eos_token_id = generation_config.eos_token_id
#             if isinstance(eos_token_id, list):
#                 eos_token_id = eos_token_id[0]
#             logger.warning(
#                 f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.'
#                 )
#             generation_config.pad_token_id = eos_token_id
#         inputs_tensor, model_input_name, model_kwargs = (self.
#             _prepare_model_inputs(inputs, generation_config.bos_token_id,
#             model_kwargs))
#         batch_size = inputs_tensor.shape[0]
#         model_kwargs['output_attentions'] = generation_config.output_attentions
#         model_kwargs['output_hidden_states'
#             ] = generation_config.output_hidden_states
#         model_kwargs['use_cache'] = generation_config.use_cache
#         accepts_attention_mask = 'attention_mask' in set(inspect.signature(
#             self.forward).parameters.keys())
#         requires_attention_mask = 'encoder_outputs' not in model_kwargs
#         if model_kwargs.get('attention_mask', None
#             ) is None and requires_attention_mask and accepts_attention_mask:
#             model_kwargs['attention_mask'
#                 ] = self._prepare_attention_mask_for_generation(inputs_tensor,
#                 generation_config.pad_token_id, generation_config.eos_token_id)
#         if not self.config.is_encoder_decoder:
#             if generation_config.pad_token_id is not None and paddle.sum(x=
#                 inputs_tensor[:, (-1)] == generation_config.pad_token_id) > 0:
#                 logger.warning(
#                     "A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer."
#                     )
#         if (self.config.is_encoder_decoder and 'encoder_outputs' not in
#             model_kwargs):
#             model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
#                 inputs_tensor, model_kwargs, model_input_name)
#         if self.config.is_encoder_decoder:
#             input_ids = self._prepare_decoder_input_ids_for_generation(
#                 batch_size, decoder_start_token_id=generation_config.
#                 decoder_start_token_id, bos_token_id=generation_config.
#                 bos_token_id, model_kwargs=model_kwargs, device=
#                 inputs_tensor.place)
#             if ('input_ids' in model_kwargs and model_input_name ==
#                 'pixel_values'):
#                 input_ids = paddle.concat(x=[input_ids, model_kwargs.pop(
#                     'input_ids')], axis=-1)
#         else:
#             input_ids = (inputs_tensor if model_input_name == 'input_ids' else
#                 model_kwargs.pop('input_ids'))
#         if streamer is not None:
#             streamer.put(input_ids.cpu())
#         input_ids_seq_length = input_ids.shape[-1]
#         has_default_max_length = kwargs.get('max_length'
#             ) is None and generation_config.max_length is not None
#         if has_default_max_length and generation_config.max_new_tokens is None:
#             warnings.warn(
#                 f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation."
#                 , UserWarning)
#         elif generation_config.max_new_tokens is not None:
#             generation_config.max_length = (generation_config.
#                 max_new_tokens + input_ids_seq_length)
#             if not has_default_max_length:
#                 logger.warn(
#                     f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)'
#                     , UserWarning)
#         if (generation_config.min_length is not None and generation_config.
#             min_length > generation_config.max_length):
#             raise ValueError(
#                 f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})'
#                 )
#         if input_ids_seq_length >= generation_config.max_length:
#             input_ids_string = ('decoder_input_ids' if self.config.
#                 is_encoder_decoder else 'input_ids')
#             logger.warning(
#                 f'Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.'
#                 )
#         is_constraint_gen_mode = (generation_config.constraints is not None or
#             generation_config.force_words_ids is not None)
#         is_contrastive_search_gen_mode = (generation_config.num_beams == 1 and
#             generation_config.top_k is not None and generation_config.top_k >
#             1 and generation_config.do_sample is False and
#             generation_config.penalty_alpha is not None and
#             generation_config.penalty_alpha > 0)
#         is_greedy_gen_mode = (generation_config.num_beams == 1 and
#             generation_config.num_beam_groups == 1 and generation_config.
#             do_sample is False and not is_constraint_gen_mode and not
#             is_contrastive_search_gen_mode)
#         is_sample_gen_mode = (generation_config.num_beams == 1 and
#             generation_config.num_beam_groups == 1 and generation_config.
#             do_sample is True and not is_constraint_gen_mode and not
#             is_contrastive_search_gen_mode)
#         is_beam_gen_mode = (generation_config.num_beams > 1 and
#             generation_config.num_beam_groups == 1 and generation_config.
#             do_sample is False and not is_constraint_gen_mode and not
#             is_contrastive_search_gen_mode)
#         is_beam_sample_gen_mode = (generation_config.num_beams > 1 and
#             generation_config.num_beam_groups == 1 and generation_config.
#             do_sample is True and not is_constraint_gen_mode and not
#             is_contrastive_search_gen_mode)
#         is_group_beam_gen_mode = (generation_config.num_beams > 1 and
#             generation_config.num_beam_groups > 1 and not
#             is_constraint_gen_mode and not is_contrastive_search_gen_mode)
#         if generation_config.num_beam_groups > generation_config.num_beams:
#             raise ValueError(
#                 '`num_beam_groups` has to be smaller or equal to `num_beams`')
#         if is_group_beam_gen_mode and generation_config.do_sample is True:
#             raise ValueError(
#                 'Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`.'
#                 )
#         if streamer is not None and generation_config.num_beams > 1:
#             raise ValueError(
#                 '`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.'
#                 )
#         if self.device.type != input_ids.device.type:
#             warnings.warn(
#                 f"You are calling .generate() with the `input_ids` being on a device type different than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model is on {self.device.type}. You may experience unexpected behaviors or slower generation. Please make sure that you have put `input_ids` to the correct device by calling for example input_ids = input_ids.to('{self.device.type}') before running `.generate()`."
#                 , UserWarning)
#         logits_processor = self._get_logits_processor(generation_config=
#             generation_config, input_ids_seq_length=input_ids_seq_length,
#             encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=
#             prefix_allowed_tokens_fn, logits_processor=logits_processor)
#         stopping_criteria = self._get_stopping_criteria(generation_config=
#             generation_config, stopping_criteria=stopping_criteria)
#         if is_greedy_gen_mode:
#             if generation_config.num_return_sequences > 1:
#                 raise ValueError(
#                     f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing greedy search.'
#                     )
#             return self.greedy_search(input_ids, logits_processor=
#                 logits_processor, stopping_criteria=stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id, eos_token_id=
#                 generation_config.eos_token_id, output_scores=
#                 generation_config.output_scores, return_dict_in_generate=
#                 generation_config.return_dict_in_generate, synced_gpus=
#                 synced_gpus, streamer=streamer, **model_kwargs)
#         elif is_contrastive_search_gen_mode:
#             if generation_config.num_return_sequences > 1:
#                 raise ValueError(
#                     f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing contrastive search.'
#                     )
#             return self.contrastive_search(input_ids, top_k=
#                 generation_config.top_k, penalty_alpha=generation_config.
#                 penalty_alpha, logits_processor=logits_processor,
#                 stopping_criteria=stopping_criteria, pad_token_id=
#                 generation_config.pad_token_id, eos_token_id=
#                 generation_config.eos_token_id, output_scores=
#                 generation_config.output_scores, return_dict_in_generate=
#                 generation_config.return_dict_in_generate, synced_gpus=
#                 synced_gpus, streamer=streamer, **model_kwargs)
#         elif is_sample_gen_mode:
#             logits_warper = self._get_logits_warper(generation_config)
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.
#                 num_return_sequences, is_encoder_decoder=self.config.
#                 is_encoder_decoder, **model_kwargs)
#             return self.sample(input_ids, logits_processor=logits_processor,
#                 logits_warper=logits_warper, stopping_criteria=
#                 stopping_criteria, pad_token_id=generation_config.
#                 pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.
#                 return_dict_in_generate, synced_gpus=synced_gpus, streamer=
#                 streamer, **model_kwargs)
#         elif is_beam_gen_mode:
#             if (generation_config.num_return_sequences > generation_config.
#                 num_beams):
#                 raise ValueError(
#                     '`num_return_sequences` has to be smaller or equal to `num_beams`.'
#                     )
#             if stopping_criteria.max_length is None:
#                 raise ValueError(
#                     '`max_length` needs to be a stopping_criteria for now.')
#             beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams
#                 =generation_config.num_beams, device=inputs_tensor.place,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.
#                 num_return_sequences, max_length=generation_config.max_length)
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.
#                 num_beams, is_encoder_decoder=self.config.
#                 is_encoder_decoder, **model_kwargs)
#             return self.beam_search(input_ids, beam_scorer,
#                 logits_processor=logits_processor, stopping_criteria=
#                 stopping_criteria, pad_token_id=generation_config.
#                 pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.
#                 return_dict_in_generate, synced_gpus=synced_gpus, **
#                 model_kwargs)
#         elif is_beam_sample_gen_mode:
#             logits_warper = self._get_logits_warper(generation_config)
#             if stopping_criteria.max_length is None:
#                 raise ValueError(
#                     '`max_length` needs to be a stopping_criteria for now.')
#             beam_scorer = BeamSearchScorer(batch_size=batch_size *
#                 generation_config.num_return_sequences, num_beams=
#                 generation_config.num_beams, device=inputs_tensor.place,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 max_length=generation_config.max_length)
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.
#                 num_beams * generation_config.num_return_sequences,
#                 is_encoder_decoder=self.config.is_encoder_decoder, **
#                 model_kwargs)
#             return self.beam_sample(input_ids, beam_scorer,
#                 logits_processor=logits_processor, logits_warper=
#                 logits_warper, stopping_criteria=stopping_criteria,
#                 pad_token_id=generation_config.pad_token_id, eos_token_id=
#                 generation_config.eos_token_id, output_scores=
#                 generation_config.output_scores, return_dict_in_generate=
#                 generation_config.return_dict_in_generate, synced_gpus=
#                 synced_gpus, **model_kwargs)
#         elif is_group_beam_gen_mode:
#             if (generation_config.num_return_sequences > generation_config.
#                 num_beams):
#                 raise ValueError(
#                     '`num_return_sequences` has to be smaller or equal to `num_beams`.'
#                     )
#             if (generation_config.num_beams % generation_config.
#                 num_beam_groups != 0):
#                 raise ValueError(
#                     '`num_beams` should be divisible by `num_beam_groups` for group beam search.'
#                     )
#             if stopping_criteria.max_length is None:
#                 raise ValueError(
#                     '`max_length` needs to be a stopping_criteria for now.')
#             has_default_typical_p = kwargs.get('typical_p'
#                 ) is None and generation_config.typical_p == 1.0
#             if not has_default_typical_p:
#                 raise ValueError(
#                     'Decoder argument `typical_p` is not supported with beam groups.'
#                     )
#             beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams
#                 =generation_config.num_beams, device=inputs_tensor.place,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.
#                 num_return_sequences, num_beam_groups=generation_config.
#                 num_beam_groups, max_length=generation_config.max_length)
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.
#                 num_beams, is_encoder_decoder=self.config.
#                 is_encoder_decoder, **model_kwargs)
#             return self.group_beam_search(input_ids, beam_scorer,
#                 logits_processor=logits_processor, stopping_criteria=
#                 stopping_criteria, pad_token_id=generation_config.
#                 pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.
#                 return_dict_in_generate, synced_gpus=synced_gpus, **
#                 model_kwargs)
#         elif is_constraint_gen_mode:
#             if (generation_config.num_return_sequences > generation_config.
#                 num_beams):
#                 raise ValueError(
#                     '`num_return_sequences` has to be smaller or equal to `num_beams`.'
#                     )
#             if stopping_criteria.max_length is None:
#                 raise ValueError(
#                     '`max_length` needs to be a stopping_criteria for now.')
#             if generation_config.num_beams <= 1:
#                 raise ValueError(
#                     '`num_beams` needs to be greater than 1 for constrained generation.'
#                     )
#             if generation_config.do_sample:
#                 raise ValueError(
#                     '`do_sample` needs to be false for constrained generation.'
#                     )
#             if (generation_config.num_beam_groups is not None and
#                 generation_config.num_beam_groups > 1):
#                 raise ValueError(
#                     '`num_beam_groups` not supported yet for constrained generation.'
#                     )
#             final_constraints = []
#             if generation_config.constraints is not None:
#                 final_constraints = generation_config.constraints
#             if generation_config.force_words_ids is not None:
#
#                 def typeerror():
#                     raise ValueError(
#                         f'`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`of positive integers, but is {generation_config.force_words_ids}.'
#                         )
#                 if not isinstance(generation_config.force_words_ids, list
#                     ) or len(generation_config.force_words_ids) == 0:
#                     typeerror()
#                 for word_ids in generation_config.force_words_ids:
#                     if isinstance(word_ids[0], list):
#                         if not isinstance(word_ids, list) or len(word_ids
#                             ) == 0:
#                             typeerror()
#                         if any(not isinstance(token_ids, list) for
#                             token_ids in word_ids):
#                             typeerror()
#                         if any(any(not isinstance(token_id, int) or
#                             token_id < 0 for token_id in token_ids) for
#                             token_ids in word_ids):
#                             typeerror()
#                         constraint = DisjunctiveConstraint(word_ids)
#                     else:
#                         if not isinstance(word_ids, list) or len(word_ids
#                             ) == 0:
#                             typeerror()
#                         if any(not isinstance(token_id, int) or token_id <
#                             0 for token_id in word_ids):
#                             typeerror()
#                         constraint = PhrasalConstraint(word_ids)
#                     final_constraints.append(constraint)
#             constrained_beam_scorer = ConstrainedBeamSearchScorer(constraints
#                 =final_constraints, batch_size=batch_size, num_beams=
#                 generation_config.num_beams, device=inputs_tensor.place,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.
#                 num_return_sequences, max_length=generation_config.max_length)
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids, expand_size=generation_config.
#                 num_beams, is_encoder_decoder=self.config.
#                 is_encoder_decoder, **model_kwargs)
#             return self.constrained_beam_search(input_ids,
#                 constrained_beam_scorer=constrained_beam_scorer,
#                 logits_processor=logits_processor, stopping_criteria=
#                 stopping_criteria, pad_token_id=generation_config.
#                 pad_token_id, eos_token_id=generation_config.eos_token_id,
#                 output_scores=generation_config.output_scores,
#                 return_dict_in_generate=generation_config.
#                 return_dict_in_generate, synced_gpus=synced_gpus, **
#                 model_kwargs)
#
#     @paddle.no_grad()
#     def contrastive_search(self, input_ids: paddle.Tensor, top_k: Optional[
#         int]=1, penalty_alpha: Optional[float]=0, logits_processor:
#         Optional[LogitsProcessorList]=None, logits_warper: Optional[
#         LogitsProcessorList]=None, stopping_criteria: Optional[
#         StoppingCriteriaList]=None, pad_token_id: Optional[int]=None,
#         eos_token_id: Optional[Union[int, List[int]]]=None,
#         output_attentions: Optional[bool]=None, output_hidden_states:
#         Optional[bool]=None, output_scores: Optional[bool]=None,
#         return_dict_in_generate: Optional[bool]=None, synced_gpus: Optional
#         [bool]=False, streamer: Optional['BaseStreamer']=None, **model_kwargs
#         ) ->Union[ContrastiveSearchOutput, paddle.Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
#         be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.contrastive_search`] directly. Use
#         generate() instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             top_k (`int`, *optional*, defaults to 1):
#                 The size of the candidate set that is used to re-rank for contrastive search
#             penalty_alpha (`float`, *optional*, defaults to 0):
#                 The degeneration penalty for contrastive search; activate when it is larger than 0
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             model_kwargs:
#                 Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
#                 If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.ContrastiveSearchDecoderOnlyOutput`], [`~generation.ContrastiveSearchEncoderDecoderOutput`]
#             or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.ContrastiveSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.ContrastiveSearchEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#         Examples:
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForCausalLM,
#         ...     StoppingCriteriaList,
#         ...     MaxLengthCriteria,
#         ... )
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
#         >>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
#         >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
#         >>> model.config.pad_token_id = model.config.eos_token_id
#         >>> input_prompt = "DeepMind Company is"
#         >>> input_ids = tokenizer(input_prompt, return_tensors="pt")
#         >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
#         >>> outputs = model.contrastive_search(
#         ...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
#         ... )
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\\n\\nIn this post, we talk about the benefits of deep learning in business and how it']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         logits_warper = (logits_warper if logits_warper is not None else
#             LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         eos_token_id_tensor = paddle.to_tensor(data=eos_token_id).to(input_ids
#             .place) if eos_token_id is not None else None
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         scores = () if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         unfinished_sequences = paddle.ones(shape=input_ids.shape[0], dtype=
#             'int64')
#         this_peer_finished = False
#         batch_size = input_ids.shape[0]
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             if model_kwargs.get('past_key_values') is None:
#                 model_kwargs['use_cache'] = True
#                 model_inputs = self.prepare_inputs_for_generation(input_ids,
#                     **model_kwargs)
#                 outputs = self(**model_inputs, return_dict=True,
#                     output_hidden_states=True, output_attentions=
#                     output_attentions)
#                 if self.config.is_encoder_decoder:
#                     last_hidden_states = outputs.decoder_hidden_states[-1]
#                 else:
#                     last_hidden_states = outputs.hidden_states[-1]
#                 logit_for_next_step = outputs.logits[:, (-1), :]
#                 model_kwargs = self._update_model_kwargs_for_generation(outputs
#                     , model_kwargs, is_encoder_decoder=self.config.
#                     is_encoder_decoder, standardize_cache_format=True)
#                 _, model_kwargs = self._expand_inputs_for_generation(
#                     expand_size=top_k, is_encoder_decoder=self.config.
#                     is_encoder_decoder, **model_kwargs)
#                 past_key_values = model_kwargs.get('past_key_values')
#                 if past_key_values is None:
#                     raise ValueError(
#                         f"{self.__class__.__name__} does not support caching and therefore **can't** be used for contrastive search."
#                         )
#                 elif not isinstance(past_key_values[0], (tuple, paddle.Tensor)
#                     ) or past_key_values[0][0].shape[0] != batch_size:
#                     raise ValueError(
#                         f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be used for contrastive search without further modifications."
#                         )
#             logit_for_next_step = logits_processor(input_ids,
#                 logit_for_next_step)
#             logit_for_next_step = logits_warper(input_ids, logit_for_next_step)
#             next_probs = paddle.nn.functional.softmax(x=logit_for_next_step,
#                 axis=-1)
#             top_k_probs, top_k_ids = paddle.topk(k=top_k, x=next_probs, axis=-1
#                 )
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += logit_for_next_step,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             new_key_values = []
#             for layer in model_kwargs['past_key_values']:
#                 items = []
#                 for item in layer:
#                     items.append(item.repeat_interleave(repeats=top_k, axis=0))
#                 new_key_values.append(items)
#             model_kwargs['past_key_values'] = new_key_values
#             """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            next_model_inputs = self.prepare_inputs_for_generation(top_k_ids
#                 .view(-1, 1), **model_kwargs)
#             outputs = self(**next_model_inputs, return_dict=True,
#                 output_hidden_states=True, output_attentions=output_attentions)
#             next_past_key_values = self._extract_past_from_model_output(outputs
#                 , standardize_cache_format=True)
#             logits = outputs.logits[:, (-1), :]
#             if self.config.is_encoder_decoder:
#                 next_hidden = outputs.decoder_hidden_states[-1]
#                 full_hidden_states = outputs.decoder_hidden_states
#             else:
#                 next_hidden = outputs.hidden_states[-1]
#                 full_hidden_states = outputs.hidden_states
#             context_hidden = last_hidden_states.repeat_interleave(repeats=
#                 top_k, axis=0)
#             selected_idx = _ranking_fast(context_hidden, next_hidden,
#                 top_k_probs, penalty_alpha, top_k)
#             next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
#             next_hidden = paddle.stack(x=paddle_aux.split(x=next_hidden.
#                 squeeze(axis=1), num_or_sections=top_k))
#             next_hidden = next_hidden[(range(batch_size)), (selected_idx), :]
#             last_hidden_states = paddle.concat(x=[last_hidden_states,
#                 next_hidden.unsqueeze(axis=1)], axis=1)
#             next_decoder_hidden_states = ()
#             for layer in full_hidden_states:
#                 layer = paddle.stack(x=paddle_aux.split(x=layer,
#                     num_or_sections=top_k))[(range(batch_size)), (
#                     selected_idx), :]
#                 next_decoder_hidden_states += layer,
#             new_key_values = ()
#             for layer in next_past_key_values:
#                 items = ()
#                 for item in layer:
#                     item = paddle.stack(x=paddle_aux.split(x=item,
#                         num_or_sections=top_k, axis=0))
#                     item = item[range(batch_size), selected_idx, ...]
#                     items += item,
#                 new_key_values += items,
#             next_past_key_values = new_key_values
#             logit_for_next_step = paddle.stack(x=paddle_aux.split(x=logits,
#                 num_or_sections=top_k))[(range(batch_size)), (selected_idx), :]
#             if self.config.is_encoder_decoder:
#                 next_step_cross_attentions = ()
#                 next_step_decoder_attentions = ()
#                 if output_attentions:
#                     for layer in outputs.cross_attentions:
#                         layer = paddle.stack(x=paddle_aux.split(x=layer,
#                             num_or_sections=top_k, axis=0))[range(
#                             batch_size), selected_idx, ...]
#                         next_step_cross_attentions += layer,
#                     for layer in outputs.decoder_attentions:
#                         layer = paddle.stack(x=paddle_aux.split(x=layer,
#                             num_or_sections=top_k, axis=0))[range(
#                             batch_size), selected_idx, ...]
#                         next_step_decoder_attentions += layer,
#                 outputs = Seq2SeqLMOutput(past_key_values=
#                     next_past_key_values, decoder_hidden_states=
#                     next_decoder_hidden_states, decoder_attentions=
#                     next_step_decoder_attentions or None, cross_attentions=
#                     next_step_cross_attentions or None)
#             else:
#                 next_step_attentions = ()
#                 if output_attentions:
#                     for layer in outputs.attentions:
#                         layer = paddle.stack(x=paddle_aux.split(x=layer,
#                             num_or_sections=top_k, axis=0))[range(
#                             batch_size), selected_idx, ...]
#                         next_step_attentions += layer,
#                 outputs = CausalLMOutputWithPast(past_key_values=
#                     next_past_key_values, hidden_states=
#                     next_decoder_hidden_states, attentions=
#                     next_step_attentions or None)
#             if synced_gpus and this_peer_finished:
#                 continue
#             if eos_token_id is not None:
#                 if pad_token_id is None:
#                     raise ValueError(
#                         'If `eos_token_id` is defined, make sure that `pad_token_id` is defined.'
#                         )
#                 next_tokens = (next_tokens * unfinished_sequences +
#                     pad_token_id * (1 - unfinished_sequences))
#             input_ids = paddle.concat(x=[input_ids, next_tokens[:, (None)]],
#                 axis=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if eos_token_id_tensor is not None:
#                 """Class Method: *.tile, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>                unfinished_sequences = unfinished_sequences.mul(next_tokens
#                     .tile(eos_token_id_tensor.shape[0], 1).not_equal(y=
#                     paddle.to_tensor(eos_token_id_tensor.unsqueeze(axis=1))
#                     ).prod(axis=0))
#             if unfinished_sequences.max() == 0 or stopping_criteria(input_ids,
#                 scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         if streamer is not None:
#             streamer.end()
#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return ContrastiveSearchEncoderDecoderOutput(sequences=
#                     input_ids, scores=scores, encoder_attentions=
#                     encoder_attentions, encoder_hidden_states=
#                     encoder_hidden_states, decoder_attentions=
#                     decoder_attentions, cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states)
#             else:
#                 return ContrastiveSearchDecoderOnlyOutput(sequences=
#                     input_ids, scores=scores, attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states)
#         else:
#             return input_ids
#
#     def greedy_search(self, input_ids: paddle.Tensor, logits_processor:
#         Optional[LogitsProcessorList]=None, stopping_criteria: Optional[
#         StoppingCriteriaList]=None, max_length: Optional[int]=None,
#         pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int,
#         List[int]]]=None, output_attentions: Optional[bool]=None,
#         output_hidden_states: Optional[bool]=None, output_scores: Optional[
#         bool]=None, return_dict_in_generate: Optional[bool]=None,
#         synced_gpus: Optional[bool]=False, streamer: Optional[
#         'BaseStreamer']=None, **model_kwargs) ->Union[GreedySearchOutput,
#         paddle.Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
#         used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
#         instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             model_kwargs:
#                 Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
#                 If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForCausalLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     StoppingCriteriaList,
#         ...     MaxLengthCriteria,
#         ... )
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
#
#         >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
#         >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
#
#         >>> input_prompt = "It might be possible to"
#         >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
#         ...     ]
#         ... )
#         >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
#
#         >>> outputs = model.greedy_search(
#         ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         eos_token_id_tensor = paddle.to_tensor(data=eos_token_id).to(input_ids
#             .place) if eos_token_id is not None else None
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         scores = () if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         unfinished_sequences = paddle.ones(shape=input_ids.shape[0], dtype=
#             'int64')
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 continue
#             next_token_logits = outputs.logits[:, (-1), :]
#             next_tokens_scores = logits_processor(input_ids, next_token_logits)
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += next_tokens_scores,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             next_tokens = paddle.argmax(x=next_tokens_scores, axis=-1)
#             if eos_token_id is not None:
#                 if pad_token_id is None:
#                     raise ValueError(
#                         'If `eos_token_id` is defined, make sure that `pad_token_id` is defined.'
#                         )
#                 next_tokens = (next_tokens * unfinished_sequences +
#                     pad_token_id * (1 - unfinished_sequences))
#             input_ids = paddle.concat(x=[input_ids, next_tokens[:, (None)]],
#                 axis=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if eos_token_id_tensor is not None:
#                 """Class Method: *.tile, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>                unfinished_sequences = unfinished_sequences.mul(next_tokens
#                     .tile(eos_token_id_tensor.shape[0], 1).not_equal(y=
#                     paddle.to_tensor(eos_token_id_tensor.unsqueeze(axis=1))
#                     ).prod(axis=0))
#             if unfinished_sequences.max() == 0 or stopping_criteria(input_ids,
#                 scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         if streamer is not None:
#             streamer.end()
#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GreedySearchEncoderDecoderOutput(sequences=input_ids,
#                     scores=scores, encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return GreedySearchDecoderOnlyOutput(sequences=input_ids,
#                     scores=scores, attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states)
#         else:
#             return input_ids
#
#     def sample(self, input_ids: paddle.Tensor, logits_processor: Optional[
#         LogitsProcessorList]=None, stopping_criteria: Optional[
#         StoppingCriteriaList]=None, logits_warper: Optional[
#         LogitsProcessorList]=None, max_length: Optional[int]=None,
#         pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int,
#         List[int]]]=None, output_attentions: Optional[bool]=None,
#         output_hidden_states: Optional[bool]=None, output_scores: Optional[
#         bool]=None, return_dict_in_generate: Optional[bool]=None,
#         synced_gpus: Optional[bool]=False, streamer: Optional[
#         'BaseStreamer']=None, **model_kwargs) ->Union[SampleOutput, paddle.
#         Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
#         For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
#             A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForCausalLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     TopKLogitsWarper,
#         ...     TemperatureLogitsWarper,
#         ...     StoppingCriteriaList,
#         ...     MaxLengthCriteria,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
#
#         >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
#         >>> model.config.pad_token_id = model.config.eos_token_id
#         >>> model.generation_config.pad_token_id = model.config.eos_token_id
#
#         >>> input_prompt = "Today is a beautiful day, and"
#         >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
#         ...     ]
#         ... )
#         >>> # instantiate logits processors
#         >>> logits_warper = LogitsProcessorList(
#         ...     [
#         ...         TopKLogitsWarper(50),
#         ...         TemperatureLogitsWarper(0.7),
#         ...     ]
#         ... )
#
#         >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
#
#         >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
#         >>> outputs = model.sample(
#         ...     input_ids,
#         ...     logits_processor=logits_processor,
#         ...     logits_warper=logits_warper,
#         ...     stopping_criteria=stopping_criteria,
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         logits_warper = (logits_warper if logits_warper is not None else
#             LogitsProcessorList())
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         eos_token_id_tensor = paddle.to_tensor(data=eos_token_id).to(input_ids
#             .place) if eos_token_id is not None else None
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         scores = () if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         unfinished_sequences = paddle.ones(shape=input_ids.shape[0], dtype=
#             'int64')
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 continue
#             next_token_logits = outputs.logits[:, (-1), :]
#             next_token_scores = logits_processor(input_ids, next_token_logits)
#             next_token_scores = logits_warper(input_ids, next_token_scores)
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += next_token_scores,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             probs = paddle.nn.functional.softmax(x=next_token_scores, axis=-1)
#             next_tokens = paddle.multinomial(x=probs, num_samples=1).squeeze(
#                 axis=1)
#             if eos_token_id is not None:
#                 if pad_token_id is None:
#                     raise ValueError(
#                         'If `eos_token_id` is defined, make sure that `pad_token_id` is defined.'
#                         )
#                 next_tokens = (next_tokens * unfinished_sequences +
#                     pad_token_id * (1 - unfinished_sequences))
#             input_ids = paddle.concat(x=[input_ids, next_tokens[:, (None)]],
#                 axis=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if eos_token_id_tensor is not None:
#                 """Class Method: *.tile, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>                unfinished_sequences = unfinished_sequences.mul(next_tokens
#                     .tile(eos_token_id_tensor.shape[0], 1).not_equal(y=
#                     paddle.to_tensor(eos_token_id_tensor.unsqueeze(axis=1))
#                     ).prod(axis=0))
#             if unfinished_sequences.max() == 0 or stopping_criteria(input_ids,
#                 scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         if streamer is not None:
#             streamer.end()
#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return SampleEncoderDecoderOutput(sequences=input_ids,
#                     scores=scores, encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return SampleDecoderOnlyOutput(sequences=input_ids, scores=
#                     scores, attentions=decoder_attentions, hidden_states=
#                     decoder_hidden_states)
#         else:
#             return input_ids
#
#     def beam_search(self, input_ids: paddle.Tensor, beam_scorer: BeamScorer,
#         logits_processor: Optional[LogitsProcessorList]=None,
#         stopping_criteria: Optional[StoppingCriteriaList]=None, max_length:
#         Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id:
#         Optional[Union[int, List[int]]]=None, output_attentions: Optional[
#         bool]=None, output_hidden_states: Optional[bool]=None,
#         output_scores: Optional[bool]=None, return_dict_in_generate:
#         Optional[bool]=None, synced_gpus: Optional[bool]=False, **model_kwargs
#         ) ->Union[BeamSearchOutput, paddle.Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
#         instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     BeamSearchScorer,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#
#         >>> # lets run beam search using 3 beams
#         >>> num_beams = 3
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(
#         ...     batch_size=1,
#         ...     num_beams=num_beams,
#         ...     device=model.device,
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
#         ...     ]
#         ... )
#
#         >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         if len(stopping_criteria) == 0:
#             warnings.warn(
#                 "You don't have defined any stopping_criteria, this will likely loop forever"
#                 , UserWarning)
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#         batch_beam_size, cur_len = input_ids.shape
#         if num_beams * batch_size != batch_beam_size:
#             raise ValueError(
#                 f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
#                 )
#         scores = () if return_dict_in_generate and output_scores else None
#         beam_indices = tuple(() for _ in range(batch_beam_size)
#             ) if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         beam_scores = paddle.zeros(shape=(batch_size, num_beams), dtype=
#             'float32')
#         beam_scores[:, 1:] = -1000000000.0
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        beam_scores = beam_scores.view((batch_size * num_beams,))
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue
#             next_token_logits = outputs.logits[:, (-1), :]
#             next_token_logits = self.adjust_logits_during_generation(
#                 next_token_logits, cur_len=cur_len)
#             next_token_scores = paddle.nn.functional.log_softmax(x=
#                 next_token_logits, axis=-1)
#             next_token_scores_processed = logits_processor(input_ids,
#                 next_token_scores)
#             next_token_scores = next_token_scores_processed + beam_scores[:,
#                 (None)].expand_as(y=next_token_scores)
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += next_token_scores_processed,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             vocab_size = next_token_scores.shape[-1]
#             """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            next_token_scores = next_token_scores.view(batch_size,
#                 num_beams * vocab_size)
#             next_token_scores, next_tokens = paddle.topk(k=2 * num_beams,
#                 largest=True, sorted=True, x=next_token_scores, axis=1)
#             next_indices = paddle.floor(paddle.divide(x=next_tokens, y=
#                 paddle.to_tensor(vocab_size)))
#             next_tokens = next_tokens % vocab_size
#             beam_outputs = beam_scorer.process(input_ids, next_token_scores,
#                 next_tokens, next_indices, pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id, beam_indices=beam_indices)
#             beam_scores = beam_outputs['next_beam_scores']
#             beam_next_tokens = beam_outputs['next_beam_tokens']
#             beam_idx = beam_outputs['next_beam_indices']
#             input_ids = paddle.concat(x=[input_ids[(beam_idx), :],
#                 beam_next_tokens.unsqueeze(axis=-1)], axis=-1)
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if model_kwargs['past_key_values'] is not None:
#                 model_kwargs['past_key_values'] = self._reorder_cache(
#                     model_kwargs['past_key_values'], beam_idx)
#             if return_dict_in_generate and output_scores:
#                 beam_indices = tuple(beam_indices[beam_idx[i]] + (beam_idx[
#                     i],) for i in range(len(beam_indices)))
#             cur_len = cur_len + 1
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         sequence_outputs = beam_scorer.finalize(input_ids, beam_scores,
#             next_tokens, next_indices, pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id, max_length=stopping_criteria.
#             max_length, beam_indices=beam_indices)
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs['sequence_scores'] = None
#             if self.config.is_encoder_decoder:
#                 return BeamSearchEncoderDecoderOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return BeamSearchDecoderOnlyOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     attentions=decoder_attentions, hidden_states=
#                     decoder_hidden_states)
#         else:
#             return sequence_outputs['sequences']
#
#     def beam_sample(self, input_ids: paddle.Tensor, beam_scorer: BeamScorer,
#         logits_processor: Optional[LogitsProcessorList]=None,
#         stopping_criteria: Optional[StoppingCriteriaList]=None,
#         logits_warper: Optional[LogitsProcessorList]=None, max_length:
#         Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id:
#         Optional[Union[int, List[int]]]=None, output_attentions: Optional[
#         bool]=None, output_hidden_states: Optional[bool]=None,
#         output_scores: Optional[bool]=None, return_dict_in_generate:
#         Optional[bool]=None, synced_gpus: Optional[bool]=False, **model_kwargs
#         ) ->Union[BeamSampleOutput, paddle.Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **beam search multinomial
#         sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.beam_sample`] directly. Use generate()
#         instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.BeamSampleDecoderOnlyOutput`], [`~generation.BeamSampleEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.BeamSampleEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     TopKLogitsWarper,
#         ...     TemperatureLogitsWarper,
#         ...     BeamSearchScorer,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#         >>> # lets run beam search using 3 beams
#         >>> num_beams = 3
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(
#         ...     batch_size=1,
#         ...     max_length=model.config.max_length,
#         ...     num_beams=num_beams,
#         ...     device=model.device,
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
#         ... )
#         >>> # instantiate logits processors
#         >>> logits_warper = LogitsProcessorList(
#         ...     [
#         ...         TopKLogitsWarper(50),
#         ...         TemperatureLogitsWarper(0.7),
#         ...     ]
#         ... )
#
#         >>> outputs = model.beam_sample(
#         ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#         batch_beam_size, cur_len = input_ids.shape
#         scores = () if return_dict_in_generate and output_scores else None
#         beam_indices = tuple(() for _ in range(batch_beam_size)
#             ) if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         beam_scores = paddle.zeros(shape=(batch_size, num_beams), dtype=
#             'float32')
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        beam_scores = beam_scores.view((batch_size * num_beams,))
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue
#             next_token_logits = outputs.logits[:, (-1), :]
#             next_token_logits = self.adjust_logits_during_generation(
#                 next_token_logits, cur_len=cur_len)
#             next_token_scores = paddle.nn.functional.log_softmax(x=
#                 next_token_logits, axis=-1)
#             next_token_scores_processed = logits_processor(input_ids,
#                 next_token_scores)
#             next_token_scores = next_token_scores_processed + beam_scores[:,
#                 (None)].expand_as(y=next_token_scores)
#             next_token_scores = logits_warper(input_ids, next_token_scores)
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += logits_warper(input_ids,
#                         next_token_scores_processed),
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             vocab_size = next_token_scores.shape[-1]
#             """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            next_token_scores = next_token_scores.view(batch_size,
#                 num_beams * vocab_size)
#             probs = paddle.nn.functional.softmax(x=next_token_scores, axis=-1)
#             next_tokens = paddle.multinomial(x=probs, num_samples=2 * num_beams
#                 )
#             next_token_scores = paddle.take_along_axis(arr=
#                 next_token_scores, axis=-1, indices=next_tokens)
#             next_token_scores, _indices = paddle.sort(descending=True, x=
#                 next_token_scores, axis=1), paddle.argsort(descending=True,
#                 x=next_token_scores, axis=1)
#             next_tokens = paddle.take_along_axis(arr=next_tokens, axis=-1,
#                 indices=_indices)
#             next_indices = paddle.floor(paddle.divide(x=next_tokens, y=
#                 paddle.to_tensor(vocab_size)))
#             next_tokens = next_tokens % vocab_size
#             beam_outputs = beam_scorer.process(input_ids, next_token_scores,
#                 next_tokens, next_indices, pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id, beam_indices=beam_indices)
#             beam_scores = beam_outputs['next_beam_scores']
#             beam_next_tokens = beam_outputs['next_beam_tokens']
#             beam_idx = beam_outputs['next_beam_indices']
#             input_ids = paddle.concat(x=[input_ids[(beam_idx), :],
#                 beam_next_tokens.unsqueeze(axis=-1)], axis=-1)
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if model_kwargs['past_key_values'] is not None:
#                 model_kwargs['past_key_values'] = self._reorder_cache(
#                     model_kwargs['past_key_values'], beam_idx)
#             if return_dict_in_generate and output_scores:
#                 beam_indices = tuple(beam_indices[beam_idx[i]] + (beam_idx[
#                     i],) for i in range(len(beam_indices)))
#             cur_len = cur_len + 1
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         sequence_outputs = beam_scorer.finalize(input_ids, beam_scores,
#             next_tokens, next_indices, pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id, max_length=stopping_criteria.
#             max_length, beam_indices=beam_indices)
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs['sequence_scores'] = None
#             if self.config.is_encoder_decoder:
#                 return BeamSampleEncoderDecoderOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return BeamSampleDecoderOnlyOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     attentions=decoder_attentions, hidden_states=
#                     decoder_hidden_states)
#         else:
#             return sequence_outputs['sequences']
#
#     def group_beam_search(self, input_ids: paddle.Tensor, beam_scorer:
#         BeamScorer, logits_processor: Optional[LogitsProcessorList]=None,
#         stopping_criteria: Optional[StoppingCriteriaList]=None, max_length:
#         Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id:
#         Optional[Union[int, List[int]]]=None, output_attentions: Optional[
#         bool]=None, output_hidden_states: Optional[bool]=None,
#         output_scores: Optional[bool]=None, return_dict_in_generate:
#         Optional[bool]=None, synced_gpus: Optional[bool]=False, **model_kwargs
#         ):
#         """
#         Generates sequences of token ids for models with a language modeling head using **diverse beam search
#         decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.group_beam_search`] directly. Use
#         generate() instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             beam_scorer (`BeamScorer`):
#                 An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#
#             model_kwargs:
#                 Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
#                 model is an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSearchDecoderOnlyOutput`] if [`~generation.BeamSearchDecoderOnlyOutput`] if
#             `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
#             [`~generation.BeamSearchEncoderDecoderOutput`] if `model.config.is_encoder_decoder=True`.
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     HammingDiversityLogitsProcessor,
#         ...     BeamSearchScorer,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#
#         >>> # lets run diverse beam search using 6 beams
#         >>> num_beams = 6
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = BeamSearchScorer(
#         ...     batch_size=1,
#         ...     max_length=model.config.max_length,
#         ...     num_beams=num_beams,
#         ...     device=model.device,
#         ...     num_beam_groups=3,
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
#         ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
#         ...     ]
#         ... )
#
#         >>> outputs = model.group_beam_search(
#         ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt bist du?']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         batch_size = len(beam_scorer._beam_hyps)
#         num_beams = beam_scorer.num_beams
#         num_beam_groups = beam_scorer.num_beam_groups
#         num_sub_beams = num_beams // num_beam_groups
#         device = input_ids.place
#         batch_beam_size, cur_len = input_ids.shape
#         if return_dict_in_generate and output_scores:
#             beam_indices = [tuple(() for _ in range(num_sub_beams *
#                 batch_size)) for _ in range(num_beam_groups)]
#         else:
#             beam_indices = None
#         if num_beams * batch_size != batch_beam_size:
#             raise ValueError(
#                 f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
#                 )
#         scores = () if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         beam_scores = paddle.full(shape=(batch_size, num_beams), fill_value
#             =-1000000000.0, dtype='float32')
#         beam_scores[:, ::num_sub_beams] = 0
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        beam_scores = beam_scores.view((batch_size * num_beams,))
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             current_tokens = paddle.zeros(shape=batch_size * num_beams,
#                 dtype=input_ids.dtype)
#             reordering_indices = paddle.zeros(shape=batch_size * num_beams,
#                 dtype='int64')
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue
#             if output_scores:
#                 processed_score = paddle.zeros_like(x=outputs.logits[:, (-1
#                     ), :])
#             for beam_group_idx in range(num_beam_groups):
#                 group_start_idx = beam_group_idx * num_sub_beams
#                 group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
#                 group_size = group_end_idx - group_start_idx
#                 batch_group_indices = []
#                 for batch_idx in range(batch_size):
#                     batch_group_indices.extend([(batch_idx * num_beams +
#                         idx) for idx in range(group_start_idx, group_end_idx)])
#                 group_input_ids = input_ids[batch_group_indices]
#                 next_token_logits = outputs.logits[(batch_group_indices), (
#                     -1), :]
#                 next_token_logits = self.adjust_logits_during_generation(
#                     next_token_logits, cur_len=cur_len)
#                 next_token_scores = paddle.nn.functional.log_softmax(x=
#                     next_token_logits, axis=-1)
#                 vocab_size = next_token_scores.shape[-1]
#                 next_token_scores_processed = logits_processor(group_input_ids,
#                     next_token_scores, current_tokens=current_tokens,
#                     beam_group_idx=beam_group_idx)
#                 next_token_scores = next_token_scores_processed + beam_scores[
#                     batch_group_indices].unsqueeze(axis=-1)
#                 next_token_scores = next_token_scores.expand_as(y=
#                     next_token_scores_processed)
#                 if output_scores:
#                     processed_score[batch_group_indices
#                         ] = next_token_scores_processed
#                 """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>                next_token_scores = next_token_scores.view(batch_size,
#                     group_size * vocab_size)
#                 next_token_scores, next_tokens = paddle.topk(k=2 *
#                     group_size, largest=True, sorted=True, x=
#                     next_token_scores, axis=1)
#                 next_indices = paddle.floor(paddle.divide(x=next_tokens, y=
#                     paddle.to_tensor(vocab_size)))
#                 next_tokens = next_tokens % vocab_size
#                 process_beam_indices = sum(beam_indices, ()
#                     ) if beam_indices is not None else None
#                 beam_outputs = beam_scorer.process(group_input_ids,
#                     next_token_scores, next_tokens, next_indices,
#                     pad_token_id=pad_token_id, eos_token_id=eos_token_id,
#                     beam_indices=process_beam_indices)
#                 beam_scores[batch_group_indices] = beam_outputs[
#                     'next_beam_scores']
#                 beam_next_tokens = beam_outputs['next_beam_tokens']
#                 beam_idx = beam_outputs['next_beam_indices']
#                 if return_dict_in_generate and output_scores:
#                     beam_indices[beam_group_idx] = tuple(beam_indices[
#                         beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in
#                         range(len(beam_indices[0])))
#                 input_ids[batch_group_indices] = group_input_ids[beam_idx]
#                 group_input_ids = paddle.concat(x=[group_input_ids[(
#                     beam_idx), :], beam_next_tokens.unsqueeze(axis=-1)],
#                     axis=-1)
#                 current_tokens[batch_group_indices] = group_input_ids[:, (-1)]
#                 reordering_indices[batch_group_indices
#                     ] = num_beams * paddle.floor(paddle.divide(x=beam_idx,
#                     y=paddle.to_tensor(group_size))
#                     ) + group_start_idx + beam_idx % group_size
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += processed_score,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             input_ids = paddle.concat(x=[input_ids, current_tokens.
#                 unsqueeze(axis=-1)], axis=-1)
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if model_kwargs['past_key_values'] is not None:
#                 model_kwargs['past_key_values'] = self._reorder_cache(
#                     model_kwargs['past_key_values'], reordering_indices)
#             cur_len = cur_len + 1
#             if beam_scorer.is_done or stopping_criteria(input_ids, scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         final_beam_indices = sum(beam_indices, ()
#             ) if beam_indices is not None else None
#         sequence_outputs = beam_scorer.finalize(input_ids, beam_scores,
#             next_tokens, next_indices, pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id, max_length=stopping_criteria.
#             max_length, beam_indices=final_beam_indices)
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs['sequence_scores'] = None
#             if self.config.is_encoder_decoder:
#                 return BeamSearchEncoderDecoderOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return BeamSearchDecoderOnlyOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     beam_indices=sequence_outputs['beam_indices'],
#                     attentions=decoder_attentions, hidden_states=
#                     decoder_hidden_states)
#         else:
#             return sequence_outputs['sequences']
#
#     def constrained_beam_search(self, input_ids: paddle.Tensor,
#         constrained_beam_scorer: ConstrainedBeamSearchScorer,
#         logits_processor: Optional[LogitsProcessorList]=None,
#         stopping_criteria: Optional[StoppingCriteriaList]=None, max_length:
#         Optional[int]=None, pad_token_id: Optional[int]=None, eos_token_id:
#         Optional[Union[int, List[int]]]=None, output_attentions: Optional[
#         bool]=None, output_hidden_states: Optional[bool]=None,
#         output_scores: Optional[bool]=None, return_dict_in_generate:
#         Optional[bool]=None, synced_gpus: Optional[bool]=None, **model_kwargs
#         ) ->Union[BeamSearchOutput, paddle.Tensor]:
#         """
#         Generates sequences of token ids for models with a language modeling head using **constrained beam search
#         decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         <Tip warning={true}>
#
#         In most cases, you do not need to call [`~generation.GenerationMixin.constrained_beam_search`] directly. Use
#         generate() instead. For an overview of generation strategies and code examples, check the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
#                 A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
#                 sorted during generation, while satisfying a list of positive constraints. For more information, the
#                 documentation of [`ConstrainedBeamSearchScorer`] should be read.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step.
#             max_length (`int`, *optional*, defaults to 20):
#                 **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
#                 tokens. The maximum length of the sequence to be generated.
#             pad_token_id (`int`, *optional*):
#                 The id of the *padding* token.
#             eos_token_id (`Union[int, List[int]]`, *optional*):
#                 The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
#             output_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more details.
#             output_hidden_states (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more details.
#             output_scores (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
#             return_dict_in_generate (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#             synced_gpus (`bool`, *optional*, defaults to `False`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
#             `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#
#
#         Examples:
#
#         ```python
#         >>> from transformers import (
#         ...     AutoTokenizer,
#         ...     AutoModelForSeq2SeqLM,
#         ...     LogitsProcessorList,
#         ...     MinLengthLogitsProcessor,
#         ...     ConstrainedBeamSearchScorer,
#         ...     PhrasalConstraint,
#         ... )
#         >>> import torch
#
#         >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
#         >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
#
#         >>> encoder_input_str = "translate English to German: How old are you?"
#         >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
#
#
#         >>> # lets run beam search using 3 beams
#         >>> num_beams = 3
#         >>> # define decoder start token ids
#         >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
#         >>> input_ids = input_ids * model.config.decoder_start_token_id
#
#         >>> # add encoder_outputs to model keyword arguments
#         >>> model_kwargs = {
#         ...     "encoder_outputs": model.get_encoder()(
#         ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
#         ...     )
#         ... }
#
#         >>> constraint_str = "Sie"
#         >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
#         >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]
#
#
#         >>> # instantiate beam scorer
#         >>> beam_scorer = ConstrainedBeamSearchScorer(
#         ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
#         ... )
#
#         >>> # instantiate logits processors
#         >>> logits_processor = LogitsProcessorList(
#         ...     [
#         ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
#         ...     ]
#         ... )
#
#         >>> outputs = model.constrained_beam_search(
#         ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
#         ... )
#
#         >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         ['Wie alt sind Sie?']
#         ```"""
#         logits_processor = (logits_processor if logits_processor is not
#             None else LogitsProcessorList())
#         stopping_criteria = (stopping_criteria if stopping_criteria is not
#             None else StoppingCriteriaList())
#         if max_length is not None:
#             warnings.warn(
#                 '`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.'
#                 , UserWarning)
#             stopping_criteria = validate_stopping_criteria(stopping_criteria,
#                 max_length)
#         if len(stopping_criteria) == 0:
#             warnings.warn(
#                 "You don't have defined any stopping_criteria, this will likely loop forever"
#                 , UserWarning)
#         pad_token_id = (pad_token_id if pad_token_id is not None else self.
#             generation_config.pad_token_id)
#         eos_token_id = (eos_token_id if eos_token_id is not None else self.
#             generation_config.eos_token_id)
#         if isinstance(eos_token_id, int):
#             eos_token_id = [eos_token_id]
#         output_scores = (output_scores if output_scores is not None else
#             self.generation_config.output_scores)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.generation_config.output_attentions)
#         output_hidden_states = (output_hidden_states if
#             output_hidden_states is not None else self.generation_config.
#             output_hidden_states)
#         return_dict_in_generate = (return_dict_in_generate if
#             return_dict_in_generate is not None else self.generation_config
#             .return_dict_in_generate)
#         scores = () if return_dict_in_generate and output_scores else None
#         decoder_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         cross_attentions = (
#             ) if return_dict_in_generate and output_attentions else None
#         decoder_hidden_states = (
#             ) if return_dict_in_generate and output_hidden_states else None
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs['encoder_outputs'].get(
#                 'attentions') if output_attentions else None
#             encoder_hidden_states = model_kwargs['encoder_outputs'].get(
#                 'hidden_states') if output_hidden_states else None
#         batch_size = len(constrained_beam_scorer._beam_hyps)
#         num_beams = constrained_beam_scorer.num_beams
#         batch_beam_size, cur_len = input_ids.shape
#         if num_beams * batch_size != batch_beam_size:
#             raise ValueError(
#                 f'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
#                 )
#         beam_scores = paddle.zeros(shape=(batch_size, num_beams), dtype=
#             'float32')
#         beam_scores[:, 1:] = -1000000000.0
#         """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>        beam_scores = beam_scores.view((batch_size * num_beams,))
#         this_peer_finished = False
#         while True:
#             if synced_gpus:
#                 this_peer_finished_flag = paddle.to_tensor(data=0.0 if
#                     this_peer_finished else 1.0).to(input_ids.place)
# >>>>>>                torch.distributed.all_reduce(this_peer_finished_flag, op=
#                     paddle.distributed.ReduceOp.SUM)
#                 if this_peer_finished_flag.item() == 0.0:
#                     break
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **
#                 model_kwargs)
#             outputs = self(**model_inputs, return_dict=True,
#                 output_attentions=output_attentions, output_hidden_states=
#                 output_hidden_states)
#             if synced_gpus and this_peer_finished:
#                 cur_len = cur_len + 1
#                 continue
#             next_token_logits = outputs.logits[:, (-1), :]
#             next_token_logits = self.adjust_logits_during_generation(
#                 next_token_logits, cur_len=cur_len)
#             next_token_scores = paddle.nn.functional.log_softmax(x=
#                 next_token_logits, axis=-1)
#             next_token_scores_processed = logits_processor(input_ids,
#                 next_token_scores)
#             next_token_scores = next_token_scores_processed + beam_scores[:,
#                 (None)].expand_as(y=next_token_scores)
#             scores_for_all_vocab = next_token_scores.clone()
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += next_token_scores,
#                 if output_attentions:
#                     decoder_attentions += (outputs.decoder_attentions,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         attentions,)
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += outputs.cross_attentions,
#                 if output_hidden_states:
#                     decoder_hidden_states += (outputs.decoder_hidden_states,
#                         ) if self.config.is_encoder_decoder else (outputs.
#                         hidden_states,)
#             vocab_size = next_token_scores.shape[-1]
#             """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            next_token_scores = next_token_scores.view(batch_size,
#                 num_beams * vocab_size)
#             next_token_scores, next_tokens = paddle.topk(k=2 * num_beams,
#                 largest=True, sorted=True, x=next_token_scores, axis=1)
#             next_indices = (next_tokens / vocab_size).astype(dtype='int64')
#             next_tokens = next_tokens % vocab_size
#             beam_outputs = constrained_beam_scorer.process(input_ids,
#                 next_token_scores, next_tokens, next_indices,
#                 scores_for_all_vocab, pad_token_id=pad_token_id,
#                 eos_token_id=eos_token_id)
#             beam_scores = beam_outputs['next_beam_scores']
#             beam_next_tokens = beam_outputs['next_beam_tokens']
#             beam_idx = beam_outputs['next_beam_indices']
#             input_ids = paddle.concat(x=[input_ids[(beam_idx), :],
#                 beam_next_tokens.unsqueeze(axis=-1)], axis=-1)
#             model_kwargs = self._update_model_kwargs_for_generation(outputs,
#                 model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
#                 )
#             if model_kwargs['past_key_values'] is not None:
#                 model_kwargs['past_key_values'] = self._reorder_cache(
#                     model_kwargs['past_key_values'], beam_idx)
#             cur_len = cur_len + 1
#             if constrained_beam_scorer.is_done or stopping_criteria(input_ids,
#                 scores):
#                 if not synced_gpus:
#                     break
#                 else:
#                     this_peer_finished = True
#         sequence_outputs = constrained_beam_scorer.finalize(input_ids,
#             beam_scores, next_tokens, next_indices, pad_token_id=
#             pad_token_id, eos_token_id=eos_token_id, max_length=
#             stopping_criteria.max_length)
#         if return_dict_in_generate:
#             if not output_scores:
#                 sequence_outputs['sequence_scores'] = None
#             if self.config.is_encoder_decoder:
#                 return BeamSearchEncoderDecoderOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions, cross_attentions
#                     =cross_attentions, decoder_hidden_states=
#                     decoder_hidden_states)
#             else:
#                 return BeamSearchDecoderOnlyOutput(sequences=
#                     sequence_outputs['sequences'], sequences_scores=
#                     sequence_outputs['sequence_scores'], scores=scores,
#                     attentions=decoder_attentions, hidden_states=
#                     decoder_hidden_states)
#         else:
#             return sequence_outputs['sequences']
#
#
# def top_k_top_p_filtering(logits: paddle.Tensor, top_k: int=0, top_p: float
#     =1.0, filter_value: float=-float('Inf'), min_tokens_to_keep: int=1
#     ) ->paddle.Tensor:
#     """
#     Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#
#     Args:
#         logits: logits distribution shape (batch size, vocabulary size)
#         top_k (`int`, *optional*, defaults to 0):
#             If > 0, only keep the top k tokens with highest probability (top-k filtering)
#         top_p (`float`, *optional*, defaults to 1.0):
#             If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
#             filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#         min_tokens_to_keep (`int`, *optional*, defaults to 1):
#             Minimumber of tokens we keep per batch example in the output.
#
#     From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#     """
#     if top_k > 0:
#         logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value,
#             min_tokens_to_keep=min_tokens_to_keep)(None, logits)
#     if 0 <= top_p <= 1.0:
#         logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value,
#             min_tokens_to_keep=min_tokens_to_keep)(None, logits)
#     return logits
#
#
# def _ranking_fast(context_hidden: paddle.Tensor, next_hidden: paddle.Tensor,
#     next_top_k_probs: paddle.Tensor, alpha: float, beam_width: int
#     ) ->paddle.Tensor:
#     """
#     Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
#     in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
#     row in the batch.
#     """
#     norm_context_hidden = context_hidden / context_hidden.norm(axis=2,
#         keepdim=True)
#     norm_next_hidden = next_hidden / next_hidden.norm(axis=2, keepdim=True)
#     x = norm_next_hidden
#     perm_1 = list(range(x.ndim))
#     perm_1[1] = 2
#     perm_1[2] = 1
#     cosine_matrix = paddle.matmul(x=norm_context_hidden, y=x.transpose(perm
#         =perm_1)).squeeze(axis=-1)
#     degeneration_penalty, _ = paddle.max(x=cosine_matrix, axis=-1
#         ), paddle.argmax(x=cosine_matrix, axis=-1)
#     """Class Method: *.view, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>    next_top_k_probs = next_top_k_probs.view(-1)
#     contrastive_score = (1.0 - alpha
#         ) * next_top_k_probs - alpha * degeneration_penalty
#     contrastive_score = paddle.stack(x=paddle_aux.split(x=contrastive_score,
#         num_or_sections=beam_width))
#     _, selected_idx = contrastive_score.max(dim=-1)
#     return selected_idx
