import sympy as sp
from sympy import symbols

import numpy as np
from numpy.linalg import inv
from numpy.polynomial.polynomial import polyvander2d, polyvander

from ezdxf.math import Vector
from typing import List, Union

from tools.general import closed, pairwise

X, Y, Z = 0, 1, 2


def diap(start, end, step=1) -> List[float]:
    """ Принимает две точки пространства и возвращает точки распределенные на заданном расстоянии
     между данными двумя.
    :param Iterable[float] start: начальная точка в пространстве
    :param Iterable[float] end: конечная точка в пространстве
    :param float step: шаг между точками
    :return: точка между start и end
    """
    start = Vector(start)
    end = Vector(end)
    d = start.distance(end)
    number_of_steps = int(d / step)
    ratio = step / d
    for i in range(number_of_steps):
        yield list(start.lerp(end, i * ratio))
    yield list(end)


def line_side(m: Vector, p1: Vector = (0, 0, 0), p2: Vector = (1, 1, 0)) -> Union['-1', '0', '1']:
    # TODO: какая то туфта - удалить
    """
    check to which side of the line (p1,p2) the point m is
    :param m: point to check
    :param p1: first point of the line
    :param p2: second point of the line
    :return: -1 if on the left side, 0 on line, 1 on the right side
    """
    p1, p2 = (p1, p2) if p1[Y] > p2[Y] else (p2, p1)
    pos = np.sign((p2[X] - p1[X]) * (m[Y] - p1[Y]) - (p2[Y] - p1[Y]) * (m[X] - p1[X]))
    return pos


def triangle_area(a, b, c, signed=False) -> float:
    # считает площадь треугольника по 3 точкам
    # TODO: вынести в elements
    area = (a[X] * (b[Y] - c[Y]) + b[X] * (c[Y] - a[Y]) + c[X] * (a[Y] - b[Y])) / 2.0
    return area if signed else abs(area)


def polygon_area(poly: List) -> float:
    # считает площадь полигона без самопересечений по точкам
    total_area = 0
    for i in range(len(poly) - 2):
        a, b, c = poly[0], poly[i + 1], poly[i + 2]
        area = triangle_area(a, b, c, True)
        total_area += area
    return abs(total_area)


def inside_polygon(p, poly: List[List[float]]) -> bool:
    # проверка точка p внутри полигона или нет
    # TODO: use cv.polygonTest
    p = [round(coord, 2) for coord in p]
    boundary_area = round(polygon_area(poly))
    partial_area = 0
    for v1, v2 in pairwise(closed(poly)):
        partial_area += triangle_area(p, v1, v2)
    if boundary_area - round(partial_area) > boundary_area * 0.01:
        return False
    return True


def inside_triangle(p, a, b, c) -> bool:
    # проверка точка p внутри треугольника или нет
    # TODO: вынести в elements
    total_area = round(triangle_area(a, b, c), 3)
    area_pab = triangle_area(p, a, b)
    area_pbc = triangle_area(p, b, c)
    area_pca = triangle_area(p, c, a)
    partial_area = round(area_pab + area_pbc + area_pca, 3)
    if partial_area == total_area:
        return True
    return False


def height_by_trigon(p=(0, 0), a=(0, 0, 0), b=(0, 0, 0), c=(0, 0, 0)):
    # расчёт высоты по трём точкам внутри которых данная p
    # TODO: вынести в elements
    axy = np.asarray(a)
    bxy = np.asarray(b)
    cxy = np.asarray(c)
    pxy = np.r_[p, 1]
    axy[Z] = 1
    bxy[Z] = 1
    cxy[Z] = 1
    area = triangle_area(axy, bxy, cxy)
    area1 = triangle_area(pxy, axy, bxy)
    area2 = triangle_area(pxy, bxy, cxy)
    area3 = triangle_area(pxy, cxy, axy)
    height = c[Z] * area1 / area + a[Z] * area2 / area + b[Z] * area3 / area
    return height


def mls2d(x, y, p=0, support_radius=1, deg: int = 1):
    """
    двумерный moving least squares

    :param x: данные по x
    :param y: данные по y соответствующие x
    :param p: опорная точка
    :param support_radius: радиус влияния
    :param deg: степень полинома
    :return: коэффициенты полинома
    """
    B = polyvander(x, deg)
    rj = np.sqrt(np.power(x - p, 2)) / support_radius
    W = np.diag(np.where(rj <= 1, 1 - 6 * rj ** 2 + 8 * rj ** 3 - 3 * rj ** 4, 0))
    c = inv(B.T @ W @ B) @ B.T @ W @ y
    return c


def mls3d(data: np.ndarray, p=(0, 0), support_radius=1, deg=(0, 0)):
    """
    трехмерный moving least squares

    :param data: набор (x,y,z) точек для аппроксимации
    :param p: опорная точка
    :param support_radius: радиус влияния
    :param deg: степень полинома
    :return: коэффициенты полинома
    """
    B = polyvander2d(data[:, 0], data[:, 1], deg)
    rj = np.sqrt(np.sum(np.power(data[:, :2] - p[:2], 2), axis=1)) / support_radius
    W = np.diag(np.where(rj <= 1, 1 - 6 * rj ** 2 + 8 * rj ** 3 - 3 * rj ** 4, 0))
    try:
        c = inv(B.T @ W @ B) @ B.T @ W @ data[:, 2]
    except np.linalg.LinAlgError:
        print('fuck')
        return np.zeros((deg[0] + 1) * (deg[0] + 1))
    return c


def rotx(angle, *, sympy=False):
    if sympy:
        angle = symbols(angle)
        return np.array([[1, 0, 0],
                         [0, sp.cos(angle), -sp.sin(angle)],
                         [0, sp.sin(angle), sp.cos(angle)]])
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])


def roty(angle, *, sympy=False):
    if sympy:
        angle = symbols(angle)
        return np.array([[sp.cos(angle), 0, sp.sin(angle)],
                         [0, 1, 0],
                         [-sp.sin(angle), 0, sp.cos(angle)]])
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


def rotz(angle, *, sympy=False):
    if sympy:
        angle = symbols(angle)
        return np.array([[sp.cos(angle), -sp.sin(angle), 0],
                         [sp.sin(angle), sp.cos(angle), 0],
                         [0, 0, 1]])
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def roteul(phi, theta, psi, *, order='XYZ', sympy=False):
    rot = {'X': rotx, 'Y': roty, 'Z': rotz}
    R1 = rot[order[0]](phi, sympy=sympy)
    R2 = rot[order[1]](theta, sympy=sympy)
    R3 = rot[order[2]](psi, sympy=sympy)
    return R1 @ R2 @ R3


def rotrpy(roll, pitch, yaw, *, order='XYZ', sympy=False):
    if order not in ['XYZ', 'ZYX']:
        raise Exception('Not a "roll pitch yaw" order')
    return roteul(yaw, pitch, roll, order=order, sympy=sympy)


def rpy2angles(R):
    pass
