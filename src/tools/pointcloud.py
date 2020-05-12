from os.path import isfile

import open3d
import numpy as np
from ezdxf.math import Vector
from numpy.polynomial.polynomial import polyval2d

from Errors import Error
from tools.math import inside_polygon, mls3d

X, Y, Z = 0, 1, 2


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
            _height_map = np.loadtxt(filename, skiprows=1, dtype=np.float32)
            _height_map = _height_map.reshape(shape)
            return _height_map
    return None


def save_height_map(height_map: np.ndarray, filename='height_map.txt'):
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
            np.savetxt(outfile, row, fmt='%-7.3f')
            outfile.write('# New row\n')


def show_height_map(height_map: np.ndarray):
    """
    Показать карту высот в виде облака точек

    :param height_map: карта высот
    """
    pcd = get_pcd_of_height_map(height_map)
    open3d.visualization.draw_geometries_with_editing([pcd])


def get_pcd_of_height_map(height_map: np.ndarray) -> open3d.geometry.PointCloud:
    """
    преобразует карту высот в open3d.PointCloud

    :param height_map: карта высот
    :return: облако точек
    """
    pcd = open3d.geometry.PointCloud()
    hm = height_map.copy().reshape(height_map.size // 3, 3)
    pcd.points = open3d.utility.Vector3dVector(hm[~np.isnan(hm).any(axis=1)])
    return pcd


def get_pcd_of_height_map_with_normals(height_map: np.ndarray, radius=None, knn=None, ref=np.array([0, 0, 1])):
    """
    преобразует карту высот в облако точек с расчитанными нормалями для каждой точки

    :param height_map: карта высот
    :param radius: в каком радиусе учитывать точки
    :param knn: максимальное количество точек для расчета
    :param ref: опорный вектор относительно которого располагать остальные нормали
    :return: облако точек
    """
    pcd = get_pcd_of_height_map(height_map)
    if radius is None and knn is None:
        raise TypeError('No radius or knn options given')
    elif knn is None:
        param = open3d.geometry.KDTreeSearchParamRadius(radius)
    elif radius is None:
        param = open3d.geometry.KDTreeSearchParamKNN(knn)
    else:
        param = open3d.geometry.KDTreeSearchParamHybrid(radius, knn)
    pcd.estimate_normals(param)
    pcd.orient_normals_to_align_with_direction(ref)
    return pcd


def get_colored_point_in_pcd(height_map: np.ndarray, idx, fg_color=(1, 0, 0), bg_color=(1, 0.706, 0)):
    """
    раскрашивает точки из idx в fg_color а всё остальное в bg_color

    :param height_map: карта высот для покраски
    :param idx: индексы flatten карты высот
    :param fg_color: цвет для выбранных точек
    :param bg_color: цвет остальных точек
    :return: облако точек
    """
    pcd = get_pcd_of_height_map(height_map)
    if idx:
        colors = np.full(np.asarray(height_map).shape, bg_color)
        for i in idx:
            colors[np.unravel_index(i, colors.shape[:2])] = fg_color
        pcd.colors = open3d.Vector3dVector(colors.reshape(colors.size // 3, 3))
    else:
        pcd.paint_uniform_color(fg_color)
    return pcd


def get_max_height_coords(height_map: np.ndarray):
    """
    возвращает координаты точки с максимальной высотой в карте высот

    :param height_map: карта высот
    :return: np.array([X,Y,Z]) координаты точки
    """
    return height_map[height_map[..., Z].argmax()]


def get_max_height_idx(height_map: np.ndarray):
    """
    возвращает индекс точки с максимальной высотой в карте

    :param height_map: карта высот
    :return: индекс размера height_map.ndim - 1
    """
    return np.unravel_index(height_map[..., Z].argmax(), height_map.shape[:2])


def get_nearest(point, height_map: np.ndarray, planar=True):
    """
    возвращает индекс точки ближайшей к данной в плоскости или в пространстве

    :param point: точка относительно которой поиск
    :param height_map: карта высот
    :param planar: в плоскости или в пространстве
    :return: индекс размера height_map.ndim - 1
    """
    if len(point) < 2 or len(point) > 3:
        raise TypeError('only 2D or 3D points')
    if planar:
        return np.unravel_index(np.nanargmin(np.sum(np.abs(height_map[..., :2] - point[:2]), axis=-1)), height_map.shape[:2])
    else:
        return np.unravel_index(np.nanargmin(np.sum(np.abs(height_map - point), axis=-1)), height_map.shape[:2])


def get_furthest(point, height_map: np.ndarray, planar=True):
    """
    возвращает индекс точки самой удалённой от данной в плоскости или в пространстве

    :param point: точка относительно которой поиск
    :param height_map: карта высот
    :param planar: в плоскости или в пространстве
    :return: индекс размера height_map.ndim - 1
    """
    if len(point) < 2 or len(point) > 3:
        raise TypeError('only 2D or 3D points')
    if planar:
        return np.unravel_index(np.sum(np.abs(height_map[..., :2] - point[:2]), axis=2).argmax(), height_map.shape[:2])
    else:
        return np.unravel_index(np.sum(np.abs(height_map[..., :] - point), axis=2).argmax(), height_map.shape[:2])


def apprx_point_height(point: Vector, height_map: np.ndarray = None, point_apprx='nearest', **kwargs) -> float:
    """
    аппроксимация высоты для данной точки p по карте высот по алгоритму
    :param point: точка для аппроксимации
    :param height_map: карта высот
    :param point_apprx: метод аппроксимации
        (default)   nearest - высота ближайшей точки к данной в облаке
                    constant - вернуть заданную высоту или 0 если не указано
                    mls - moving least squares аппроксимация
    :param kwargs: параметры для каджого из методов
                    constant:
                        height - высота которую надо вернуть (default - 0)
                    mls:
                        support_radius - радиус в пределах которого учитывать точки (default - 1.)
                        degree - порядок полинома для аппроксимации в формате (xm, ym) (default - (1,1))
    :return: аппроксимированная высота точки
    """
    ind = ((0, 0, -1, -1), (0, -1, -1, 0))
    if height_map is None and point_apprx != 'constant':
        raise Error('cannot approximate height without point cloud. use constant height or provide cloud')
    if point_apprx == 'constant':
        return kwargs.get('height', 0)
    elif inside_polygon(point, height_map[ind][:, :2]):
        if point_apprx == 'mls':
            return mls_height_apprx(height_map, point, **kwargs)
        elif point_apprx == 'nearest':
            idx_first = get_nearest(point, height_map, True)
            first = Vector(height_map[idx_first])
            return first.z
        else:
            raise Error(f'no such algorithm {point_apprx}')
    else:
        print(f'point {point} not in the area')
        return 0


def mls_height_apprx(pointcloud, point, support_radius = 1., degree = (1, 1)) -> float:
    """
    аппроксимация высоты точки по облаку с помощью moving least squares

    :param pointcloud: карта высот
    :param point: точка для аппроксимации
    :param kwargs:
        support_radius - радиус в пределах которого учитывать точки (default - 1.)
        degree - порядок полинома для аппроксимации в формате (xm, ym) (default - (1,1))
    :return:
    """
    data = pointcloud[~np.isnan(pointcloud).any(axis=-1)].reshape(-1, 3)
    cond = np.sum(np.power(data[:, :2] - point[:2], 2), axis=-1) <= support_radius ** 2
    data = data[cond] - (point[X], point[Y], 0)
    if data.size >= ((degree[0] + 1) * (degree[1] + 1)):
        c = mls3d(data, (0, 0), support_radius, degree).reshape((degree[0] + 1, degree[1] + 1))
        z = polyval2d(0, 0, c)
        return z
    elif data.size > 0:
        return np.mean(data)
    else:
        return 0
