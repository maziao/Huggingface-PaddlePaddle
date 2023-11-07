from utils.registry import Registry, build_from_config, RegistryList
ACTIVATION = Registry('activation')
ATTENTION = Registry('attention')
DECODER_BLOCK = Registry('decoder_block')
EMBEDDING = Registry('embedding')
ENCODER_BLOCK = Registry('encoder_block')
MLP = Registry('mlp')
MODEL_HEAD = Registry('model_head')
ENCODER_ONLY_MODEL = Registry('encoder_only_model')
DECODER_ONLY_MODEL = Registry('decoder_only_model')
ENCODER_DECODER_MODEL = Registry('encoder_decoder_model')
LM_HEAD_MODEL = Registry('lm_head_model')
CLS_HEAD_MODEL = Registry('cls_head_model')
DOUBLE_HEAD_MODEL = Registry('double_head_model')
CRITERION = Registry('criterion')
MODULE = RegistryList(name='module', registries=[ACTIVATION, ATTENTION,
    DECODER_BLOCK, EMBEDDING, ENCODER_BLOCK, MLP, MODEL_HEAD,
    ENCODER_ONLY_MODEL, DECODER_ONLY_MODEL, ENCODER_DECODER_MODEL,
    LM_HEAD_MODEL, CLS_HEAD_MODEL, DOUBLE_HEAD_MODEL, CRITERION])


def build_activation(config):
    return build_from_config(config, ACTIVATION)


def build_attention(config):
    return build_from_config(config, ATTENTION)


def build_decoder_block(config):
    return build_from_config(config, DECODER_BLOCK)


def build_embedding(config):
    return build_from_config(config, EMBEDDING)


def build_encoder_block(config):
    return build_from_config(config, ENCODER_BLOCK)


def build_mlp(config):
    return build_from_config(config, MLP)


def build_model_head(config):
    return build_from_config(config, MODEL_HEAD)


def build_encoder_only_model(config):
    return build_from_config(config, ENCODER_ONLY_MODEL)


def build_decoder_only_model(config):
    return build_from_config(config, DECODER_ONLY_MODEL)


def build_encoder_decoder_model(config):
    return build_from_config(config, ENCODER_ONLY_MODEL)


def build_lm_head_model(config):
    return build_from_config(config, LM_HEAD_MODEL)


def build_cls_head_model(config):
    return build_from_config(config, CLS_HEAD_MODEL)


def build_double_head_model(config):
    return build_from_config(config, DOUBLE_HEAD_MODEL)
