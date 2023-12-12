import yaml
import os.path
import inspect
import logging

from config.base import BaseConfig
from modules.registry import MODULE
from dataset.registry import NLP_DATASET

logger = logging.getLogger(__name__)


class AutoConfig:

    @classmethod
    def from_yaml(cls, config_path: str, model_name: str = None):
        with open(config_path) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        """Param priority: model-specific params > global params > common params"""
        global_params = dict()
        if 'global_params' in configuration.keys():
            global_params.update(configuration['global_params'])
        if model_name is not None:
            global_params.update(configuration['models'][model_name])

        return cls.build_config_recursively(configuration['arch'], global_params)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str) -> BaseConfig:
        with open(os.path.join(pretrained_model_path, 'config.yaml')) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        return cls.build_config_recursively(configuration['arch'])

    @classmethod
    def build_config_recursively(cls, config, global_params: dict = None) -> BaseConfig:
        """Build a config instance recursively.

        Args:
            config: config dict, should have fields 'type' and 'args' ('args' can be omitted in some cases).
            global_params: global param dict, overwrite all other occurrences.

        Returns:
            An instance of `class BaseConfig`.
        """
        if global_params is None:
            global_params = dict()

        """Try to acquire module class from registry."""
        # TODO: Combine all modules for easier searching
        module_class = MODULE.get_module(config['type'])
        if module_class is None:
            module_class = NLP_DATASET.get(config['type'])
        if module_class is None:
            logger.error(f"Module {config['type']} is not defined.")
            raise NotImplementedError(f"Module {config['type']} is not defined.")

        """Try to acquire config class for the module."""
        try:
            config_class = module_class.config_class
        except AttributeError:
            logger.error(f"Module {module_class} does not have attribute `config_class`.")
            raise AttributeError(
                f"Module {module_class} does not have attribute `config_class`."
            )

        """Go through signature of config_class, find valid value for each parameter."""
        if 'args' not in config:
            config['args'] = {}
        kwargs = dict(type=config['type'])

        for name, param in inspect.signature(config_class.__init__).parameters.items():
            if name == 'self' or name == 'type':
                continue

            # The field appeared either in common config or in global config.
            if name in config['args'] or name in global_params:
                if name in config['args'] and name in global_params:
                    if config['args'][name] is not None:
                        logger.warn(f"Param `{name}` in {config_class.__name__} appeared both in global and local "
                                    f"config, but got different non-empty values ({global_params[name]} and "
                                    f"{config['args'][name]}). Adopt global value ({global_params[name]}).")
                    value = global_params[name]
                elif name in config['args'] and name not in global_params:
                    value = config['args'][name]
                else:
                    logger.info(
                        f"Param `{name}` in {config_class.__name__} did not appear in local config, adopt global "
                        f"value ({global_params[name]}).")
                    value = global_params[name]
                # The field corresponds to another config instance, build config recursively.
                if isinstance(value, dict):
                    if 'type' in value:
                        value = cls.build_config_recursively(value, global_params)
                    else:
                        raise ValueError(f"Config for module '{name}' should at least have a field 'type', but only got"
                                         f" {value.keys()}.")
                kwargs[name] = value
            # The field did not appear in config file.
            else:
                # The field must be specified (no default value in config_class).
                if param.default == inspect.Parameter.empty:
                    logger.error(f"Param `{name}` in {config_class.__name__} did not appear in global or local config,"
                                 f" and did not have a default value. Please specify its value.")
                    raise TypeError(f"Param `{name}` in {config_class.__name__} did not appear in global or local "
                                    f"config, and did not have a default value. Please specify its value.")
                # Adopt default value in config_class.
                else:
                    logger.info(f"Param `{name}` in {config_class.__name__} did not appear in global or local config, "
                                f"adopt default value ({param.default}).")

        try:
            return config_class(**kwargs)
        except TypeError:
            raise TypeError(
                f"error occurred when initializing `class {config_class.__name__}`, expected kwargs: "
                f"{inspect.signature(config_class.__init__)}, but got {kwargs} instead."
            )


if __name__ == '__main__':
    cfg = AutoConfig.from_yaml('model_config/opt.yaml', model_name='opt-125m')
    print(cfg)
