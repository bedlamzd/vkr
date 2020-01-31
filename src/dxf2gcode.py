"""
dxf2gcode.py
Author: bedlamzd of MT.lab

Чтение .dxf файла, слайсер этого dxf, перевод в трёхмерное пространство на рельеф объектов,
генерация gcode в соответствующий файл
"""
import ezdxf as ez
from typing import Union, Optional
from gcodeGen import *
from elements import *
from utilities import print_objects
import globalValues
from globalValues import get_settings_values, settings_sections
from scanner import find_cookies
from cookie import Cookie

# TODO: написать логи
DEFAULT_F0 = 2500  # mm/minute
DEFAULT_F1 = 1000  # mm/minute
DEFAULT_TABLE_HEIGHT = 30  # mm


def gcode_generator(dwg: Drawing, cookies: Optional[List[Cookie]] = None,
                    height_map: Optional[np.ndarray] = None,
                    extrusion_coefficient: float = 0.041,
                    extrusion_multiplex: float = 1,
                    p0: float = 0.05, p1: float = 0.05, p2: float = 0.05,
                    z_offset: float = 0.2,
                    **kwargs):
    """
    Принимает обработанный рисунок, положения объектов и рельеф и генерирует из этого Gcode.
    Сохраняет Gcode по завершению.

    :param dwg: Рисунок для проекции на рельеф
    :param cookies: положение объектов в области
    :param height_map: карта высот области
    :param extrusion_coefficient: коэффициент подачи глазури
    :param extrusion_multiplex: множитель подачи для начала контуров
    :param p0: относительная длина линии до которой экструзия повышена
    :param p1: относительная длина линии до которой после p0 нет экструзии
    :param p2: относительная длина линии до которой экструзия обычная
    0 <= p0 <= p1 <= p2 <= 1
    :param z_offset: постоянное смещение от поверхности объекта
    :param kwargs: дополнительные параметры
    :keyword table_height: рабочая максимальная высота
    :keyword F0: скорость быстрых перемещений
    :keyword F1: скорость медленных перемещений (при печати)
    :return: None
    """
    # TODO: дописать под работу для генерации тестового gcode (печать по плоскости на расстоянии от стола)
    # TODO: дописать под генерацию НЕдинамической подачи
    filename = kwargs.get('filename', 'cookie.gcode')
    Z_max = kwargs.get('table_height', DEFAULT_TABLE_HEIGHT)
    F0 = kwargs.get('F0', DEFAULT_F0)
    F1 = kwargs.get('F1', DEFAULT_F1)
    point_apprx = kwargs.get('point_apprx', 'mls')
    E = 0
    gcode = Gcode() + kwargs.get('pre_gcode', [])
    gcode += home()
    gcode += set_position(E=0)
    gcode += move_Z(Z_max, F0)
    for count, cookie in enumerate(cookies, 1):
        Z_up = cookie.max_height + 5 if cookie.max_height + 5 <= Z_max else Z_max
        gcode += gcode_comment(f'{count:3d} cookie')
        dwg.center = cookie.center[:2]
        dwg.rotation = cookie.rotation
        dwg.add_z(height_map or cookie.height_map, point_apprx=point_apprx, height=kwargs.get('height', 0))
        for layer_index, layer in enumerate(sorted(dwg.layers.values(), key=lambda x: x.priority)):
            gcode += gcode_comment(f'{layer_index:3d} layer: {layer.name} in drawing')
            if layer.name == 'Contour':  # or layer.priority == 0:
                gcode += gcode_comment(f'    layer skiped')
                print(f'Layer skipped. Name: {layer.name}; Priority: {layer.priority}')
                continue
            for contour_index, contour in enumerate(layer.contours):
                printed_length = 0
                dE = extrusion_coefficient * extrusion_multiplex
                gcode += gcode_comment(f'    {contour_index:3d} contour in layer')
                gcode += linear_move(X=contour.first_point.x, Y=contour.first_point.y, Z=Z_up)
                gcode += move_Z(contour.first_point.z + z_offset)
                gcode += linear_move('G1', F=F1)
                last_point = contour.first_point
                for element_index, element in enumerate(contour.elements, 1):
                    gcode += gcode_comment(f'        {element_index:3d} element in contour')
                    for point in element.get_points()[1:]:
                        dL = point.distance(last_point)
                        E += round(dE * dL, 3)
                        gcode += linear_move('G1', X=point.x, Y=point.y, Z=point.z + z_offset, E=E)
                        printed_length += dL
                        last_point = point
                        printed_percent = printed_length / contour.length
                        if printed_percent < p0:
                            dE = extrusion_coefficient * extrusion_multiplex
                        elif printed_percent < p1:
                            dE = 0
                        elif printed_percent < p2:
                            dE = extrusion_coefficient
                        else:
                            dE = 0
                gcode += linear_move(F=F0)
                gcode += move_Z(Z_up)
    gcode += move_Z(Z_max)
    gcode += home()
    print('Команды сгенерированы')
    gcode.save(filename)


def test_gcode(path_to_dxf, d_e=0.041, height=0, center=(0, 0), rotation=0, *, dynamic_extrusion=True, **kwargs):
    """
    Функция генерирующая тестовый gcode на плоскости
    для дополнительных параметров смотреть gcode_generator

    :param path_to_dxf: путь до рисунка
    :param d_e: коэффициент подачи глазури
    :param height: высота от стола на которой печатать
    :param center: центр печеньки как (Х, Y)
    :param rotation: угол поворота печеньки в радианах
    параметры доступные только через ключевые слова
    :keyword dynamic_extrusion: динамическая подача или нет (изменяется ли коэффициент подачи в процессе)
    :param kwargs:
        :keyword filename: имя файла тестового gcode (default - planar_test.gcode)
        :keyword p0: относительная длина первого участка линии на котором подача равна k * d_e
        :keyword p1: относительная длина участка линии на котором подача равна 0 (за исключением первого участка,
                     если он не ноль)
        :keyword p2: относительная длина участка линии на котором подача равна d_e (за исключением второго участка)
                     после этого и до конца линии подача 0
        :keyword extrusion_multiplex: коэффициент усиления подачи на первом участке, (default p2/p1)
    :return: None
    """
    filename = kwargs.pop('filename', 'planar_test.gcode')
    if dynamic_extrusion:
        p0 = kwargs.get('p0', 0.05)
        p1 = kwargs.get('p1', 0.15)
        p2 = kwargs.get('p2', 0.9)
        k = kwargs.get('extrusion_multiplex', p1 / p0)
    else:
        k = 1
        p0 = 0
        p1 = 0
        p2 = 0
    dxf = ez.readfile(path_to_dxf)
    dwg = Drawing(dxf)
    dwg.slice()
    preGcode = kwargs.get('pre_gcode') or ['G0 E1 F300', 'G92 E0', 'G0 F3000']
    cookie = Cookie()
    cookie._center = (*center, 0)
    cookie._rotation = rotation
    cookie._max_height = height
    gcode_generator(dwg, [cookie], None, d_e, k, p0, p1, p2, pre_gcode=preGcode, point_apprx='constant',
                    height=height, filename=filename, **kwargs)


def dxf2gcode(path_to_dxf: str, *args, **kwargs):
    """
    Функция обработки dxf в Gcode

    :param path_to_dxf: путь до рисунка
    :return: None
    """

    settings_values = get_settings_values(**{k: settings_sections[k] for k in settings_sections if
                                             settings_sections[k][0] in ['GCoder', 'Table']})
    kwargs = {**settings_values, **kwargs}

    # прочесть dxf
    dxf = ez.readfile(path_to_dxf)
    dwg = Drawing(dxf)
    dwg.slice(kwargs.get('slice_step'))
    print(dwg)
    if globalValues.height_map is None:
        height_map = globalValues.read_height_map()
    else:
        height_map = globalValues.height_map
    if globalValues.cookies is None:
        cookies, _ = find_cookies('height_map.png', height_map)  # найти положения объектов на столе
        if len(cookies) != 0:
            print_objects(cookies, f'Объектов найдено: {len(cookies):{3}}')
            print()
    else:
        cookies = globalValues.cookies
    gcode_generator(dwg, cookies, height_map, **kwargs)
