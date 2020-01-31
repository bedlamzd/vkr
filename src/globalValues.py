from math import radians
from numpy import loadtxt, float32, ndarray, savetxt
from os.path import isfile
import configparser

config_path = 'settings.ini'


def string2list(sep=' ', f=None):
    """
    декоратор для применения заранее заданной функции f к элементам списка полученных из строки с помощью wrapper

    :param sep: разделитель строки
    :param f: функция для элементов
    :return: враппер преобразующий строку в список
    """

    def _string2list(string: str, s=' ', func=None):
        return string.split(s) if func is None else [func(x) for x in string.split(s)]

    def wrapper(string):
        return _string2list(string, sep, f)

    return wrapper


def get_settings_values(path=config_path, **kwargs):
    """
    считать значения из конфига применив к ним соответствующую функцию форматирования

    :param path: путь до конфига
    :param kwargs: словарь с параметрами для получения
    :return:
    """
    settings_values = {}
    for setting, (section, formatting) in kwargs.items():
        value = get_setting(path, section, setting)
        settings_values[setting] = formatting(value) if formatting is not None else value
    return settings_values


def read_height_map(filename='height_map.txt'):
    """
    функция чтения сохранённой карты высот

    :param filename: файл с данными
    :return: numpy.ndarray карту высот размеров указанных в файле
    """
    if isfile(filename):
        with open(filename, 'r') as infile:
            print('Читаю карту высот')
            shape = infile.readline()
            shape = shape[1:-2]
            shape = tuple(i for i in map(int, shape.split(', ')))
            _height_map = loadtxt(filename, skiprows=1, dtype=float32)
            _height_map = _height_map.reshape(shape)
            return _height_map
    return None


def save_height_map(height_map: ndarray, filename='height_map.txt'):
    """
    сохранить карту высот как .txt файл

    :param height_map: карта высот
    :param filename: название файла
    :return: None
    """
    with open(filename, 'w') as outfile:
        outfile.write('{0}\n'.format(height_map.shape))  # записать форму массива для обратного преобразования
        outfile.write('# Data starts here\n')  # обозначить где начинаются данные
        for row in height_map:  # последовально для каждого ряда сохранить данные из него в файл
            savetxt(outfile, row, fmt='%-7.3f')
            outfile.write('# New row\n')


def get_config(path):
    """
    Выбираем файл настроек
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


def update_setting(path, section, setting, value):
    """
    Обновляем параметр в настройках
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)


def get_setting(path, section, setting) -> str:
    """
    Выводим значение из настроек
    """
    config = get_config(path)
    value = config.get(section, setting)
    msg = "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value
    )
    print(msg)
    return value


height_map = None  # переменная хранения карты высот
cookies = None  # переменная хранения обнаруженных объектов
none_handler = lambda func: lambda str: None if str == '' else func(str)  # хэндлер пустых строк из конфига

# словарь всех параметров в конфиге и соответствующих им форматирований для получения данных в правильном виде, а не
# как строки
settings_sections = {
    'table_width': ('Table', none_handler(float)),
    'table_length': ('Table', none_handler(float)),
    'table_height': ('Table', none_handler(float)),
    'x0': ('Table', none_handler(float)),
    'y0': ('Table', none_handler(float)),
    'z0': ('Table', none_handler(float)),
    'pixel_size': ('Camera', none_handler(float)),
    'focal_length': ('Camera', none_handler(float)),
    'camera_angle': ('Camera', none_handler(lambda x: radians(float(x)))),
    'camera_angle_2': ('Camera', none_handler(lambda x: radians(float(x)))),
    'camera_angle_3': ('Camera', none_handler(lambda x: float(x))),
    'camera_height': ('Camera', none_handler(float)),
    'camera_shift': ('Camera', none_handler(float)),
    'distance_camera2laser': ('Camera', none_handler(float)),
    'ref_height': ('Scanner', none_handler(float)),
    'ref_width': ('Scanner', none_handler(float)),
    'ref_gap': ('Scanner', none_handler(float)),
    'ref_n': ('Scanner', none_handler(float)),
    'roi': ('Scanner', none_handler(string2list(', ', int))),
    'reverse': ('Scanner', lambda str: True if str == 'True' else False),
    'mirrored': ('Scanner', lambda str: True if str == 'True' else False),
    'extraction_mode': ('Scanner', None),
    'avg_time': ('Scanner', none_handler(float)),
    'laser_angle_tol': ('Scanner', none_handler(float)),
    'laser_pos_tol': ('Scanner', none_handler(float)),
    'accuracy': ('GCoder', none_handler(float)),
    'z_offset': ('GCoder', none_handler(float)),
    'extrusion_coefficient': ('GCoder', none_handler(float)),
    'retract_amount': ('GCoder', none_handler(float)),
    'p0': ('GCoder', none_handler(float)),
    'p1': ('GCoder', none_handler(float)),
    'p2': ('GCoder', none_handler(float)),
    'slice_step': ('GCoder', none_handler(float)),
}
