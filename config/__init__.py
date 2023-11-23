import yaml
import os.path
import inspect
import logging
from models.registry import MODULE
from dataset.registry import NLP_DATASET
from config.base import BaseConfig

logger = logging.getLogger(__name__)


class AutoConfig:

    @classmethod
    def from_yaml(cls, config_path: str, model_name: str = None):
        with open(config_path) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        global_params = dict()
        if 'global_params' in configuration.keys():
            global_params.update(configuration['global_params'])
        if model_name is not None:
            global_params.update(configuration['models'][model_name])
        return cls.build_config_recursively(configuration['arch'], global_params)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        with open(os.path.join(pretrained_model_path, 'config.yaml')) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        return cls.build_config_recursively(configuration['arch'])

    @classmethod
    def build_config_recursively(cls, config, global_params: dict = None):
        if global_params is None:
            global_params = dict()

        module_class = MODULE.get_module(config['type'])
        if module_class is None:
            module_class = NLP_DATASET.get(config['type'])

        if module_class is None:
            logger.error(f"Module {config['type']} is not defined.")
            raise NotImplementedError(f"Module {config['type']} is not defined.")
        try:
            config_class = module_class.config_class
        except AttributeError:
            logger.error(f"Module {module_class} does not have attribute `config_class`.")
            raise AttributeError(
                f"Module {module_class} does not have attribute `config_class`."
            )

        kwargs = dict(type=config['type'])
        for name, param in inspect.signature(config_class.__init__).parameters.items():
            if name == 'self' or name == 'type':
                continue

            if name in config['args']:
                value = config['args'][name]
                if isinstance(value, dict):
                    if 'type' in value and 'args' in value:
                        value = cls.build_config_recursively(value, global_params)
                if name in global_params:
                    if value is not None and value != global_params[name]:
                        logger.warn(f"Param `{name}` in {config_class} appeared both in global and local config, but "
                                    f"got different non-empty values ({global_params[name]} and "
                                    f"{config['args'][name]}). Adopt global value ({global_params[name]}).")
                    value = global_params[name]
                kwargs[name] = value
            elif name in global_params:
                kwargs[name] = global_params[name]
                logger.info(f"Param `{name}` in {config_class} did not appear in local config, adopt global value "
                            f"({global_params[name]}).")
            else:
                if param.default == inspect.Parameter.empty:
                    logger.error(f"Param `{name}` in {config_class} did not appear in global or local config, and did "
                                 f"not have a default value. Please specify its value.")
                    raise TypeError(f"Param `{name}` in {config_class} did not appear in global or local config, "
                                    f"and did not have a default value. Please specify its value.")
                else:
                    logger.info(f"Param `{name}` in {config_class} did not appear in global or local config, adopt "
                                f"default value ({param.default}).")

        try:
            return config_class(**kwargs)
        except TypeError:
            raise TypeError(
                f"error occurred when initializing `class {config_class}`, expected kwargs: "
                f"{inspect.signature(config_class.__init__)}, but got {kwargs} instead."
            )


if __name__ == '__main__':
    from utils.registry import build_from_config
    from models.registry import LM_HEAD_MODEL

    # cfg = Config.from_yaml('model_config/gpt2.yaml', model_name='gpt2-tiny')
    cfg = AutoConfig.from_yaml('model_config/bloom.yaml', model_name='bloom-560m')
    print(cfg)
    cfg.save_as_yaml('data.yaml')
    # model = build_from_config(cfg, LM_HEAD_MODEL)
    # print(model)
