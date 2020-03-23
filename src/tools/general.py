from functools import reduce
from itertools import tee
import numpy as np
from typing import Any, Optional

def nothing(*args, **kwargs):
    pass

def normalize(img: np.ndarray, value=1) -> np.ndarray:
    array = img.copy().astype(np.float64)
    array = (array - array.min()) / (array.max() - array.min()) * value
    return array

def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def closed(iterable):
    """ ABCD -> A, B, C, D, A """
    return [item for item in iterable] + [iterable[0]]


def avg(*arg) -> float:
    """
     Вычисляет среднее арифметическое
    :param arg: набор чисел
    :return: среднее
    """
    return reduce(lambda a, b: a + b, arg) / len(arg)


def update_existing_keys(old: dict, new: dict):
    return {k: new[k] if k in new else old[k] for k in old}


def print_objects(objects: Any, pre_msg: Optional[str] = None, object_msg: Optional[str] = '',
                  sep: Optional[str] = '#'):
    """
    Печатает коллекцию объектов поддерживающих функцию print с каким то однотипным сообщением перед каждым
    :param objects: коллекция объектов
    :param pre_msg: сообщение в самом начале
    :param object_msg: сообщение под каждым объектом
    :param sep: разделитель объектов
    :return: None
    """
    if pre_msg is not None:
        print(pre_msg)
    print(sep * 30)
    for count, obj in enumerate(objects, 1):
        print(f'{object_msg}', f'№{count:3d}')
        print(sep * 30)
        print(obj)
        print(sep * 30)
