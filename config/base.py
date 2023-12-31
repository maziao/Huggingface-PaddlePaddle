import os.path

import yaml
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """
    Base class for all config classes, provide basic methods for serialization, saving, etc.
    """
    type: str

    def __str__(self):
        lines = []
        for key, module in self.__dict__.items():
            mod_str = str(module)
            mod_str = self._add_indent(mod_str, 2)
            lines.append('(' + key + '): ' + mod_str)
        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    @staticmethod
    def _add_indent(s_, indent: int = 2):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(indent * ' ' + line) for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def to_dict(self):
        config = self._to_dict_recursively()
        return {'arch': config}

    def _to_dict_recursively(self):
        config = dict()
        for key, value in self.__dict__.items():
            if key != 'type':
                if isinstance(value, BaseConfig):
                    config[key] = value._to_dict_recursively()
                else:
                    config[key] = value
        return {
            'type': self.__dict__['type'],
            'args': config
        }

    def save_as_yaml(self, path):
        config = self.to_dict()
        with open(path, 'w+') as file:
            yaml.dump(config, file, sort_keys=False)

    def save_pretrained(self, path):
        self.save_as_yaml(os.path.join(path, 'config.yaml'))


@dataclass
class PseudoConfig(BaseConfig):
    """
    A pseudo config class for those classes who do not have any parameters.
    """
    pass
