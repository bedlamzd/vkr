from typing import Optional, List, Union


def abstract_gcode_command(code: str = None, **kwargs) -> str:
    """
    basic function to construct gcode commands
    :param string code: gcode command code
    :param kwargs: rest of the corresponding command params as strings
    :return string command:
    """
    if code is None:
        raise Exception('Gcode command code should be specified.')
    command = f'{code} '
    for key, value in kwargs.items():
        if key != 'code' and value is not None:
            command += f'{key}{value} '
    return command


def gcode_comment(comment: str = '') -> str:
    return ';' + comment


def linear_move(code: str = 'G0', **kwargs) -> str:
    """
    Create a G0 or G1 command
    :param str code: rapid (default) or slow movement
    :keyword float X: x coordinate to move to
    :keyword float Y: y coordinate to move to
    :keyword float Z: z coordinate to move to
    :keyword float E: e coordinate to move to (extrusion amount)
    :keyword float F: feed rate (head speed)
    :return str command: gcode command as a string
    """
    if code not in ['G0', 'G1']:
        raise Exception('code should be G0 or G1')
    params = {'X': None, 'Y': None, 'Z': None, 'E': None, 'F': None}
    params = {k: f'{kwargs[k]:3.3f}' if k in kwargs and kwargs[k] is not None else None for k in params}
    params['code'] = code
    return abstract_gcode_command(**params)


def set_position(*args, **kwargs) -> str:
    code = 'G92'
    params = {'X': None, 'Y': None, 'Z': None, 'E': None}
    params = {k: f'{kwargs[k]:3.3f}' if k in kwargs and kwargs[k] is not None else None for k in params}
    return abstract_gcode_command(code=code, **params)


def home(*args, **kwargs) -> str:
    """
    Homing command for the printer.
    :param args: could contain 'O', 'X', 'Y', 'Z' as additional parameters described below
    str O: if position is known then no homing
    str X: only home X
    str Y: only home Y
    str Z: only home Z
    e.g.: G28('X', 'Z') -> 'G28 X Z'
    :keyword Optional[float] R: raise on R before homing
    :return str command:
    """
    code = 'G28'
    params = {'O': None, 'R': None, 'X': None, 'Y': None, 'Z': None}
    params = {k: f'{kwargs[k]:3.3f}' if k in kwargs and kwargs[k] is not None else None for k in params}
    return abstract_gcode_command(code=code, **params)


move_Z = lambda z, f=None: linear_move(Z=z, F=f)


def write_gcode(gcode_instructions: Union[List[str], 'Gcode'], filename: str = 'cookie.gcode'):
    """
    Генерирует текстовый файл с инструкциями для принтера

    :param List[str] gcode_instructions: массив строк с командами для принтера
    :param str filename: имя файла для записи команд
    """
    with open(filename, 'w+') as gcode:
        for line in gcode_instructions:
            gcode.write(line + '\n')
    print('Команды для принтера сохранены.')


class Gcode:
    def __init__(self, instructions: Optional[List[str]] = None, name: str = 'cookie.gcode'):
        self.name = name
        self.instructions = instructions if instructions else []  # type: List[str]

    def save(self, filename='cookie.gcode'):
        write_gcode(self.instructions, filename=filename)

    def __str__(self) -> str:
        return f'Gcode {self.name}. Contains {self.__len__()} instructions.'

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, item: int) -> str:
        if item < self.__len__():
            return self.instructions[item]
        else:
            raise IndexError('list index out of range')

    def __iter__(self) -> str:
        for command in self.instructions:
            yield command

    def __add__(self, other: Union[List[str], str]):
        """adds commands at the end of the instructions"""
        if isinstance(other, str):
            instructions = self.instructions + [other]
        elif isinstance(other, List):
            if all(isinstance(item, str) for item in other):
                instructions = self.instructions + other
            else:
                raise TypeError('list should contain only strings')
        elif isinstance(other, Gcode):
            instructions = self.instructions + other.instructions
        else:
            raise TypeError(f'cannot add {type(other)} to Gcode')
        return Gcode(instructions)

    def __radd__(self, other):
        """adds commands at the beginning of the instructions"""
        if isinstance(other, str):
            instructions = [other] + self.instructions
        elif isinstance(other, List):
            if all(isinstance(item, str) for item in other):
                instructions = other + self.instructions
            else:
                raise TypeError('list should contain only strings')
        elif isinstance(other, Gcode):
            instructions = other.instructions + self.instructions
        else:
            raise TypeError(f'cannot add {type(other)} to Gcode')
        return Gcode(instructions)
