from utils.registry import Registry, build_from_config

NLP_DATASET = Registry('nlp_dataset')


def build_dataset(config):
    return build_from_config(config, NLP_DATASET)
