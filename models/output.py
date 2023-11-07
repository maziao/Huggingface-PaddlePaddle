import paddle
from typing import OrderedDict, Any, Tuple


class _FIELD_BASE:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


_FIELD = _FIELD_BASE('_FIELD')
_FIELDS = '__dataclass_fields__'


def fields(class_or_instance):
    """Return a tuple describing the fields of this dataclass.

    Accepts a dataclass or an instance of one. Tuple elements are of
    type Field.
    """
    try:
        fields = getattr(class_or_instance, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance')
    return tuple(f for f in fields.values() if f._field_type is _FIELD)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)
        if not len(class_fields):
            raise ValueError(f'{self.__class__.__name__} has no fields.')
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(
                f'{self.__class__.__name__} should not have more than one required field.'
                )
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for
            field in class_fields[1:])
        if other_fields_are_none and not isinstance(paddle.Tensor, first_field
            ):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or not len(
                        element) == 2 or not isinstance(element[0], str):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:
                            raise ValueError(
                                f'Cannot set key/value for {element}. It needs to be a tuple (key, value).'
                                )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.'
            )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.'
            )

    def pop(self, *args, **kwargs):
        raise Exception(
            f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        raise Exception(
            f'You cannot use ``update`` on a {self.__class__.__name__} instance.'
            )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
