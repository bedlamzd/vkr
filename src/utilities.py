"""
utilities.py
Author: bedlamzd of MT.lab

Различные функции необходимые в процессе работы
"""

from typing import List, Union, Tuple, Iterable, Sequence, Optional, Any, TypeVar
from itertools import tee
from functools import reduce, wraps
import numpy as np
from numpy.linalg import inv
from numpy.polynomial.polynomial import polyvander, polyvander2d, polyval2d
import cv2
import open3d
import imutils
from numpy import arctan, sqrt, tan, arccos, pi
from ezdxf.math.vector import Vector

""" Some tools for convenience """

# TODO: рефактор и комментарии

X, Y, Z = 0, 1, 2  # переменные для более явного обращения к координатам в массиве вместо цифр

T = TypeVar('T')


def show_img(img, winname='image', exit_key=27):
    cv2.namedWindow(winname)  # создать окно
    cv2.imshow(winname, img)  # показать в окне картинку
    # пока окно открыто и кнопка выхода не нажата ждать
    while cv2.getWindowProperty(winname, 0) >= 0 and cv2.waitKey(50) != exit_key:
        pass
    cv2.destroyAllWindows()  # завершить процессы


def normalize(img, value=1):
    array = img.copy().astype(np.float64)
    array = (array - array.min()) / (array.max() - array.min()) * value
    return array


def show_height_map(height_map: np.ndarray):
    """
    Показать карту высот в виде облака точек

    :param height_map: карта высот
    """
    pcd = get_pcd_of_height_map(height_map)
    open3d.draw_geometries_with_editing([pcd])


def get_pcd_of_height_map(height_map: np.ndarray) -> open3d.geometry.PointCloud:
    """
    преобразует карту высот в open3d.PointCloud

    :param height_map: карта высот
    :return: облако точек
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.Vector3dVector(height_map.copy().reshape(height_map.size // 3, 3))
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
        raise
    elif knn is None:
        param = open3d.geometry.KDTreeSearchParamRadius(radius)
    elif radius is None:
        param = open3d.geometry.KDTreeSearchParamKNN(knn)
    else:
        param = open3d.geometry.KDTreeSearchParamHybrid(radius, knn)
    open3d.geometry.estimate_normals(pcd, param)
    open3d.geometry.orient_normals_to_align_with_direction(pcd, ref)
    return pcd


def find_contours(img: Union[np.ndarray, str]) -> (List, np.ndarray):
    """
    Находит контуры в изображении

    :param img: картинка либо путь до картинки
    :return: возвращает контуры из изображения и изображение с отмеченными на нём контурами
    """
    # проверка параметр строка или нет
    original = None
    gray = None
    if isinstance(img, str):
        original = cv2.imread(img)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            original = img.copy()
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            original = cv2.merge((img.copy(), img.copy(), img.copy()))
            gray = img.copy()
    else:
        raise TypeError(f'передан {type(img)}, ожидалось str или numpy.ndarray')

    # избавление от минимальных шумов с помощью гауссова фильтра и отсу трешхолда
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, gausThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # нахождение замкнутых объектов на картинке с помощью морфологических алгоритмов
    kernel = np.ones((5, 5), np.uint8)
    # gausThresh = cv2.morphologyEx(gausThresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(gausThresh, cv2.MORPH_OPEN, kernel, iterations=10)
    # найти однозначный задний фон
    sureBg = cv2.dilate(opening, kernel, iterations=3)
    distTrans = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    # однозначно объекты
    ret, sureFg = cv2.threshold(distTrans, 0.1 * distTrans.max(), 255, 0)
    sureFg = np.uint8(sureFg)
    # область в которой находятся контура
    unknown = cv2.subtract(sureBg, sureFg)
    # назначение маркеров
    ret, markers = cv2.connectedComponents(sureFg)
    # отмечаем всё так, чтобы у заднего фона было точно 1
    markers += 1
    # помечаем граничную область нулём
    markers[unknown == 255] = 0
    markers = cv2.watershed(original, markers)
    # выделяем контуры на изображении
    original[markers == -1] = [0, 0, 255]
    # вырезаем ненужный контур всей картинки
    contours = []
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker] = 255
        tmp = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        tmp = imutils.grab_contours(tmp)
        contour = sorted(tmp, key=cv2.contourArea, reverse=True)[0]
        contours.append(contour)
    blankSpace = np.full(gray.shape, 255, dtype='uint8')
    blankSpace[markers <= 1] = 0
    # применяем на изначальную картинку маску с задним фоном
    result = cv2.bitwise_and(original, original, mask=blankSpace)
    for contour in contours:
        cv2.drawContours(result, [contour], -1, np.random.randint(0, 255, 3).tolist(), 1)
        # show_img(result)
    return contours, result


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
        return np.unravel_index(np.sum(np.abs(height_map[..., :2] - point[:2]), axis=2).argmin(), height_map.shape[:2])
    else:
        return np.unravel_index(np.sum(np.abs(height_map[..., :] - point), axis=2).argmin(), height_map.shape[:2])


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


def mid_idx(arr: Sequence[T], shift: Optional[int] = 0) -> Union[T, int]:
    """
    находит средний элемент(ы) в массиве и его индекс(ы)

    :param arr: массив для поиска
    :param shift: сдвиг от центра
    :return: элемент массива и его индекс если len(arr)%2==1 или слайс двух средних элементов и индекс первого из них
    """
    mid = int(len(arr) / 2) + shift
    if len(arr) % 2 == 0:
        return arr[mid - 1: mid + 1], mid - 1
    else:
        return arr[mid], mid


def find_center_and_rotation(contour, rotation=True):
    """
    Найти центр и поворот контура

    :param contour: контур для расчётв
    :param rotation: находить ли его поворот
    :return:
        центр контура (row, col)
        центр контура (row, col) и его поворот
    """
    moments = cv2.moments(contour)
    center_x = moments['m10'] / moments['m00']  # row
    center_y = moments['m01'] / moments['m00']  # column
    center = (center_x, center_y)
    if rotation:
        a = moments['m20'] / moments['m00'] - center_x ** 2
        b = 2 * (moments['m11'] / moments['m00'] - center_x * center_y)
        c = moments['m02'] / moments['m00'] - center_y ** 2
        theta = 1 / 2 * np.arctan(b / (a - c)) + (a < c) * pi / 2
        return center, theta
    else:
        return center


def pairwise(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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


def closed(iterable):
    """ ABCD -> A, B, C, D, A """
    return [item for item in iterable] + [iterable[0]]


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


def avg(*arg) -> float:
    """
     Вычисляет среднее арифметическое
    :param arg: набор чисел
    :return: среднее
    """
    return reduce(lambda a, b: a + b, arg) / len(arg)


def distance(p1, p2: Union[List[float], Tuple[float]] = (.0, .0, .0), simple=False) -> float:
    # TODO: вынести в elements
    """
        Calculate distance between 2 points either 2D or 3D
        :param list of float p1: точка от которой считается расстояние
        :param list of float p2: точка до которой считается расстояние
        :param bool simple: if True то оценочный расчёт расстояния без использования sqrt
    """
    p1 = list(p1)
    p2 = list(p2)
    if len(p1) < 3:
        p1.append(0)
    if len(p2) < 3:
        p2.append(0)
    if simple:
        return abs(p1[X] - p2[X]) + abs(p1[Y] - p2[Y]) + abs(p1[Z] - p2[Z])
    return sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2 + (p1[Z] - p2[Z]) ** 2)


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


def mls_height_apprx(height_map, point, **kwargs) -> float:
    """
    аппроксимация высоты точки по облаку с помощью moving least squares

    :param height_map: карта высот
    :param point: точка для аппроксимации
    :param kwargs:
        support_radius - радиус в пределах которого учитывать точки (default - 1.)
        degree - порядок полинома для аппроксимации в формате (xm, ym) (default - (1,1))
    :return:
    """
    sup_r = kwargs.get('support_radius', 1.)
    deg = kwargs.get('degree', (1, 1))
    data = height_map.reshape(height_map.size // 3, 3)
    cond = np.sum(np.power(data[:, :2] - point[:2], 2), axis=1) <= sup_r ** 2
    data = data[cond] - (point[X], point[Y], 0)
    if data.size >= ((deg[0] + 1) * (deg[1] + 1)):
        c = mls3d(data, (0, 0), sup_r, deg).reshape((deg[0] + 1, deg[1] + 1))
        z = polyval2d(0, 0, c)
        return z
    else:
        return 0


def update_existing_keys(old: dict, new: dict):
    return {k: new[k] if k in new else old[k] for k in old}


def find_angle_of_view(view_range: int = 640,
                       focal: float = 2.9,
                       pxl_size: float = 0.005) -> float:
    """
    расчитывает угол обзора камеры

    :param view_range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxl_size: размер пикселя на матрице
    :return: угол в радианах
    """
    return 2 * arctan(view_range * pxl_size / 2 / focal)


def find_camera_angle(view_width: float,
                      frame_width: int = 640,
                      camera_height: float = 150,
                      focal: float = 2.9,
                      pxl_size: float = 0.005) -> float:
    """
    Вычисление угла наклона камеры от вертикали по ширине обзора камеры и
    её высоте над поверхностью.

    :param view_width: ширина обзора по центру кадра в мм
    :param frame_width: ширина кадра в пикселях
    :param camera_height: высота камеры над поверхностью в мм
    :param focal: фокусное расстояние линзы
    :param pxl_size: размер пикселя на матрице в мм
    :return: угол наклона камеры в радианах
    """
    view_angle = find_angle_of_view(frame_width, focal, pxl_size)
    cos_camera_angle = 2 * camera_height / view_width * tan(view_angle / 2)
    camera_angle = arccos(cos_camera_angle)
    return camera_angle


def find_camera_angle2(img: np.ndarray,
                       distance_camera2laser: float = 90,
                       camera_height: float = 150,
                       focal: float = 2.9,
                       pixel_size: float = 0.005,
                       **kwargs) -> Tuple[float, np.ndarray]:
    """
    Вычисление угла наклона камеры от вертикали по расстоянию до лазера,
    высоте над поверхностью стола и изображению положения лазера с камеры.

    :param np.ndarray img: цветное или чб изображение (лучше чб)
    :param float distance_camera2laser: расстояние от камеры до лазера в мм
    :param float camera_height: расстояние от камеры до поверхности в мм
    :param float focal: фокусное расстояние камеры в мм
    :param float pixel_size: размер пикселя на матрице в мм
    :param kwargs: дополнительные параметры для обработки изображения
    :keyword ksize: int размер окна для ф-ии гаусса, нечетное число
    :keyword sigma: float сигма для ф-ии гаусса
    :raises Exception: 'Dimensions are not correct' if img not 3D or 2D
    :return: угол камеры в радианах
    """
    import cv2
    from scanner import predict_laser, laplace_of_gauss
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        raise Exception('Dimensions are not correct')
    roi = kwargs.get('roi', ((0, gray.shape[0]), (0, gray.shape[1])))
    gray = gray[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    deriv = laplace_of_gauss(gray, kwargs.get('ksize', 29), kwargs.get('sigma', 4.45))
    laser = predict_laser(deriv, 0, img.shape[0])
    middle, idx = mid_idx(laser)
    middle = avg(laser[idx])
    img[laser.astype(int) + roi[0][0], [i + roi[1][0] for i in range(laser.size)]] = (0, 255, 0)
    cv2.circle(img, (idx + roi[1][0], int(middle) + roi[0][0]), 2, (0, 0, 255), -1)
    dp0 = (middle - img.shape[0] / 2 - 1) * pixel_size
    tan_gamma = dp0 / focal
    d2h_ratio = distance_camera2laser / camera_height
    tan_alpha = (d2h_ratio - tan_gamma) / (1 + d2h_ratio * tan_gamma)
    camera_angle = arctan(tan_alpha)
    return camera_angle, img


def angle_from_video(video: Union[str, int], **kwargs):
    """
    Имплементация find_camera_angle2 для видео вместо отдельного снимка

    :param video: путь к видео или номер девайса для захвата
    :param kwargs: параметры функции find_camera_angle2
    :keyword distance_camera2laser: расстояние от камеры до лазера в мм
    :keyword camera_height: расстояние от камеры до поверхности в мм
    :keyword focal: фокусное расстояние камеры в мм
    :keyword pixel_size: размер пикселя на матрице в мм
    :keyword ksize: int размер окна для ф-ии гаусса, нечетное число
    :keyword sigma: float сигма для ф-ии гаусса
    :return: средний угол за все обработанные кадры
    """
    import cv2
    cap = cv2.VideoCapture(video)
    mean_angle = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('video ended or crashed')
            break
        angle, frame = find_camera_angle2(frame, **kwargs)
        count += 1
        mean_angle = (mean_angle * (count - 1) + angle) / count
        msg = f'Количество измерений: {count}\n' + \
              f'Средний угол: {np.rad2deg(mean_angle):4.2f}\n' + \
              f'Угол в текущем кадре: {np.rad2deg(angle):4.2f}'
        print(msg, '#' * 30, sep='\n')
        cv2.imshow('frame', frame)
        ch = cv2.waitKey(15)
        if ch == 27:
            print('closed by user')
            break
    cap.release()
    cv2.destroyAllWindows()
    return mean_angle


def select_hsv_values(video):
    """
    функция помощник для подбора hsv значений для фильтра
    :param video: видеопоток либо с камеры либо из файла
    :return:
    """
    params = {'h1': 0, 'h2': 255, 's1': 0, 's2': 255, 'v1': 0, 'v2': 255}
    setwin = 'hsv_set'
    reswin = 'result'
    cv2.namedWindow(setwin, cv2.WINDOW_NORMAL)
    cv2.namedWindow(reswin, cv2.WINDOW_NORMAL)
    for key in params:
        cv2.createTrackbar(key, setwin, params[key], 255, nothing)
    cv2.createTrackbar('mask', setwin, 0, 1, nothing)

    # noinspection PyShadowingNames
    def get_params(win='hsv_set'):
        for key in params:
            params[key] = int(cv2.getTrackbarPos(key, win))
        m = int(cv2.getTrackbarPos('mask', setwin))
        hsv_lower = tuple(params[k] for k in ['h1', 's1', 'v1'])
        hsv_upper = tuple(params[k] for k in ['h2', 's2', 'v2'])
        return hsv_lower, hsv_upper, m

    cap = cv2.VideoCapture(video)
    lowerb, upperb = (0, 0, 0), (255, 255, 255)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lowerb, upperb, m = get_params(setwin)
            hsv = cv2.inRange(hsv, lowerb, upperb, None)
            result = cv2.bitwise_and(frame, frame, mask=hsv) if m == 1 else hsv
            cv2.imshow(reswin, result)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ch = cv2.waitKey(15)
        if ch == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return lowerb, upperb


def nothing(*args, **kwargs):
    pass


class Error(Exception):
    """
    класс для ошибок внутри моего кода и алгоритмов
    """

    def __init__(self, msg=None):
        self.message = f'Unknown error' if msg is None else msg


def decor_stream2img(img_func):
    """
    Декоратор позволяющий использовать функции для кадров с видео

    :param img_func: функция работающая только с кадрами
    :return: генератор
    """
    import cv2
    @wraps(img_func)
    def wrapper(video, loops=False, *args, **kwargs):
        """
        Принимает на вход видео к которому покадрово нужно применить функцию

        :param video: видео для обработки
        :param loops: зациклить видео или нет (если video это поток с камеры то False в любом случае)
        :param args: параметры для функции
        :param kwargs: доп параметры и именные аргументы для функции
        :keyword max_loops: максимальное число циклов по видео. default = 10. None - бесконечно
        :return: поочерёдно результат img_func для каждого кадра
        """
        count_loops = 0
        max_loops = kwargs.pop('max_loops', 10)
        cap = cv2.VideoCapture(video)
        if isinstance(video, int):
            loops = False
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                res = img_func(frame, *args, **kwargs)
                yield res
            elif loops:
                if max_loops is None or count_loops < max_loops:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    count_loops += 1
                else:
                    print('max loops reached')
                    cap.release()
            else:
                print('video ended or crashed')
                cap.release()

    return wrapper


def height_from_stream(video):
    import cv2
    from scanner import laplace_of_gauss, predict_laser, predict_zero_level, find_coords
    params = {'distance_camera2laser': (900, 1500),
              'camera_angle': (300, 900),
              'Ynull': (0, 480),
              'Yend': (480, 480),
              'Xnull': (0, 640),
              'Xend': (640, 640),
              }
    setwin = 'settings'

    cv2.namedWindow(setwin)
    for key, (initial, max_val) in params.items():
        cv2.createTrackbar(key, setwin, initial, max_val, nothing)

    def get_params(par: dict, win=setwin):
        par_vals = {}
        for key in par.keys():
            par_vals[key] = cv2.getTrackbarPos(key, win)
        return par_vals

    K = 29
    SIGMA = 4.45
    THRESH = 5

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        values = get_params(params)
        camera_angle = values.get('camera_angle', 300) / 10 * np.pi / 180
        values['camera_angle'] = camera_angle
        d = values.get('distance_camera2laser', 900) / 10
        values['distance_camera2laser'] = d
        row_start = values.pop('Ynull')
        row_stop = values.pop('Yend')
        col_start = values.pop('Xnull')
        col_stop = values.pop('Xend')
        if row_start >= row_stop and col_start >= col_stop:
            roi = frame
        else:
            roi = frame[row_start:row_stop, col_start:col_stop]
        if ret is False:
            cap.release()
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (K, K), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, mask = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)
        der = laplace_of_gauss(gray, K, SIGMA)
        der = cv2.bitwise_and(der, der, mask=mask)
        laser = predict_laser(der, row_start, row_stop)
        laser = np.pad(laser, (col_start, frame.shape[1] - col_stop), mode='constant')
        zero_lvl, _ = predict_zero_level(laser, frame.shape[0] // 2)
        zero_lvl[(zero_lvl < row_start) | (zero_lvl > row_stop)] = row_stop - 1
        laser[laser > zero_lvl] = zero_lvl[laser > zero_lvl]
        coords = find_coords(0, laser, zero_lvl, frame.shape,
                             focal_length=2.9,
                             pixel_size=0.005,
                             table_height=100,
                             camera_shift=113,
                             **values)
        height = coords[:, Z]
        frame[:, frame.shape[1] // 2 - 1] = (0, 255, 0)
        for column in range(col_start, col_stop):
            row = laser[column]
            frame[int(row), column] = (0, 255, 0)
            frame[int(zero_lvl[column]), column] = (255, 0, 0)
        frame[[row_start, row_stop - 1], col_start:col_stop] = (127, 127, 0)
        frame[row_start:row_stop, [col_start, col_stop - 1]] = (127, 127, 0)
        max_column = laser[col_start:col_stop].argmax() + col_start
        max_row = int(laser[max_column])
        cv2.circle(frame, (max_column, max_row), 3, (0, 0, 255), -1)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        ch = cv2.waitKey(15)
        print(coords[col_start:col_stop, Z].max())
        if ch == 27:
            cap.release()
            continue
    cap.release()
    cv2.destroyAllWindows()


def apprx_x(coords: np.ndarray, height_map: np.ndarray, d, y, h, w, tol):
    # TODO: доделать и внедрить в find_coords или scanning желательно отдельной функцией
    n = 0
    checkpoint = 0
    p_idx = np.abs(coords[:, Y] - y).argmin()
    point = coords[p_idx]
    if abs(point[Y] - y) < w / 2 and abs(point[Z] - h) < tol:
        n += 1
        if n == 1:
            checkpoint = p_idx
        else:
            x = np.array([d * ((n - 2) + i / (p_idx - checkpoint)) for i in range(p_idx - checkpoint)])
            height_map[checkpoint:p_idx, :, X] = x
    return height_map


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


def tnc(x, poly: np.poly1d):
    p1 = poly.deriv(1)
    p2 = poly.deriv(2)
    tangent = p1(x)
    normal = -1 / tangent
    curv = p2(x) / (p1(x) ^ 2 + 1) ** 1.5
    return tangent, normal, curv


def iggm(data: np.ndarray, points: np.ndarray, **kwargs):
    # черновик для внедрения improved grey gravity method
    rho_min = kwargs.get('rho_min', 1)
    s = kwargs.get('s', 3)
    rho_max = kwargs.get('rho_max', s * rho_min)
    w_min = kwargs.get('w_min', 1)
    w_max = kwargs.get('w_max', 1)
    l = kwargs.get('l', 30)
    support = kwargs.get('supporting_radius', 10)
    deg = kwargs.get('deg', 3)
    new_centers = np.array(data.shape[1])
    for point in points:
        poly = np.poly1d(mls2d(data, point, support, 3)[::-1])
        t, n, rho = tnc(point, poly)
        if rho > rho_max:
            rho = rho_max
        elif rho < rho_min:
            rho = rho_min
        w = (rho - rho_min) / (rho_max - rho_min) * w_max + (rho - rho_max) / (rho_min - rho_max) * w_min


def camera_calibration(video,  # видео с которого брать кадры
                       grid,  # размеры сетки
                       *, delay=0.5,  # задержка между снимками
                       samples=10,  # количество необходимых кадров
                       win_size=(5, 5),  # не помню
                       zero_zone=(-1, -1),  # не помню
                       square_size=1,  # размер квадрата сетки
                       criteria=None):  # критерии для субпиксельных углов шахматки
    import time
    mtx, rvecs, tvecs, dist, newcameramtx, roi = None, None, None, None, None, None
    once = False
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    points = np.zeros((grid[0] * grid[1], 3), np.float32)
    points[:, :2] = np.mgrid[:grid[0], :grid[1]].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cv2.namedWindow("Video")
    focus = int(min(cap.get(cv2.CAP_PROP_FOCUS) * 100, 2 ** 31 - 1))
    cv2.createTrackbar("Focus", "Video", focus, 1000, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_shape = (frame_width, frame_height)
    good_samples = 0
    last_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if good_samples < samples:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, grid)
                if ret:
                    cur_time = time.time()
                    dif = abs(last_time - cur_time)
                    if dif >= delay:
                        last_time = time.time()
                        good_samples += 1
                        objpoints.append(points)
                        corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                        imgpoints.append(corners)
                        print(f'sample taken. current samples: {good_samples}')
                cv2.drawChessboardCorners(frame, grid, corners, ret)
            elif not once:
                once = not once
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_shape, None, None)
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, frame_shape, 1, frame_shape)
                print(f'camera matrix:      \n{mtx}\n'
                      f'new camera matrix:  \n{newcameramtx}\n'
                      f'distortion coef:    \n{dist}\n'
                      f'radial vectors:     \n{rvecs}\n'
                      f'tangential vectors: \n{tvecs}\n'
                      f'roi:                \n{roi}')
                print(f'camera intrinsic:\n{mtx}')
            else:
                dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
                cv2.imshow('undistorted', dst)
            cv2.imshow('frame', frame)
        if cv2.waitKey(15) == 27:
            cap.release()
    cap.release()
    cv2.destroyAllWindows()
    # TODO: сохранение mtx и etc. в файл
    return mtx, newcameramtx, dist, rvecs, tvecs, roi


def find_camera_pose(rvec, tvec):
    R = cv2.Rodrigues(rvec)[0].T  # матрица поворота мира относительно камеры
    pos = -R @ tvec  # координаты камеры относительно центра мира
    roll = np.arctan2(-R[2, 1], R[2, 2])
    pitch = np.arcsin(R[2, 0])
    yaw = np.arctan2(-R[1, 0], R[0, 0])
    return pos.reshape(3, ), np.array((roll, pitch, yaw))


def camera_pose_img(img, mtx,  # изображение с шахматкой и матрица параметров камеры
                    grid, square_size=1, first_corner_coord=(0, 0),
                    # размер сетки, размер клетки, координата первого угла
                    win_size=(5, 5), zero_zone=(-1, -1), criteria=None,  # параметры поиска углов на изображении
                    draw=False, rotate=False):  # отрисовывать результат или нет
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_180)
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    objp = np.zeros((7 * 7, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[:7, :7].T.reshape(-1, 2) * square_size + first_corner_coord
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, grid)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
        if draw:
            cv2.drawChessboardCorners(img, grid, corners, ret)
        ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, None)
        (coords, angles) = find_camera_pose(rvec, tvec) if ret else (None, None)
    else:
        coords, angles = None, None
    return coords, angles, img if draw else coords, angles


def generate_chessboard(square_size=30, grid=(8, 8)):
    chessboard = np.full(np.multiply(square_size, grid), 255, dtype=np.uint8)
    for m in range(grid[0]):
        row_start = square_size * m
        row_stop = square_size * (m + 1)
        for n in range(grid[1]):
            col_start = square_size * (2 * n + (m % 2))
            col_stop = col_start + square_size
            chessboard[row_start:row_stop, col_start:col_stop] = 0
    return chessboard


def find_pose_video(video, mtx, grid=(8, 8), **kwargs):
    # TODO: после калибровки прописать углы и расстояния в конфиг автоматически
    gen = decor_stream2img(camera_pose_img)
    cv2.imshow('chb', generate_chessboard(75, grid))
    cv2.waitKey(500)
    for res in gen(video, mtx=mtx, grid=(grid[0] - 1, grid[1] - 1), draw=True, **kwargs):
        if res[0] is not None:
            print(f'X: {res[0][X]:4.2f} units, {np.rad2deg(res[1][X]):4.1f} grad\n'
                  f'Y: {res[0][Y]:4.2f} units, {np.rad2deg(res[1][Y]):4.1f} grad\n'
                  f'Z: {-res[0][Z]:4.2f} units, {np.rad2deg(res[1][Z]):4.1f} grad\n')
        cv2.imshow('res', res[2])
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
