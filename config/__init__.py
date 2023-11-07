import yaml
import inspect
from models.registry import MODULE
from config.base import BaseConfig


class Config:

    @classmethod
    def from_yaml(cls, config_path, model_name: str=None):
        with open(config_path) as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        if model_name is not None:
            global_params = configuration['models'][model_name]
        else:
            global_params = None
        return cls.build_config_recursively(configuration['arch'],
            global_params)

    @classmethod
    def build_config_recursively(cls, config, global_params: dict=None):
        module_class = MODULE.get_module(config['type'])
        if module_class is None:
            raise NotImplementedError(f'Module {module_class} is not defined.')
        try:
            config_class = module_class.config_class
        except AttributeError:
            raise AttributeError(
                f'Module {module_class} does not have attribute `config_class`.'
                )
        kwargs = dict(type=config['type'])
        if 'args' in config and config['args'] is not None:
            for key, value in config['args'].items():
                if isinstance(value, dict):
                    if 'type' in value and 'args' in value:
                        value = cls.build_config_recursively(value,
                            global_params)
                if key in global_params:
                    value = global_params[key]
                kwargs[key] = value
        try:
            return config_class(**kwargs)
        except TypeError:
            raise TypeError(
                f"error occurred when initializing `class {config['type']}`, expected kwargs: {inspect.signature(config_class.__init__)}, but got {kwargs} instead."
                )


if __name__ == '__main__':
    from utils.registry import build_from_config
    from models.registry import LM_HEAD_MODEL
    cfg = Config.from_yaml('model_config/temp.yaml', model_name='gpt2-tiny')
    print(cfg)
    model = build_from_config(cfg, LM_HEAD_MODEL)
    print(model)
