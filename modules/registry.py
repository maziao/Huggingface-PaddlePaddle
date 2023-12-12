from typing import Any
from utils.registry import build_from_config, RegistryList
from modules.activation import ACTIVATION
from modules.attention import ATTENTION
from modules.block import EMBEDDING_BLOCK, ENCODER_BLOCK, DECODER_BLOCK
from modules.criterion import CRITERION
from modules.embedding import EMBEDDING
from modules.head import MODEL_HEAD
from modules.mlp import MLP
from modules.model import (
    ENCODER_ONLY_MODEL,
    DECODER_ONLY_MODEL,
    ENCODER_DECODER_MODEL,
    LM_HEAD_MODEL,
    CLS_HEAD_MODEL,
    DOUBLE_HEAD_MODEL
)
from modules.norm import NORM

MODULE = RegistryList(name='module', registries=[
    ACTIVATION,
    ATTENTION,
    EMBEDDING_BLOCK, ENCODER_BLOCK, DECODER_BLOCK,
    CRITERION,
    EMBEDDING,
    MODEL_HEAD,
    MLP,
    ENCODER_ONLY_MODEL, DECODER_ONLY_MODEL, ENCODER_DECODER_MODEL, LM_HEAD_MODEL, CLS_HEAD_MODEL, DOUBLE_HEAD_MODEL,
    NORM
])


def build_module(config: Any):
    return build_from_config(config, MODULE)
