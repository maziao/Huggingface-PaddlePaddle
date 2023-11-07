from dataclasses import dataclass


@dataclass
class BaseConfig:
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
    def _add_indent(s_, indent: int=2):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(indent * ' ' + line) for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s
