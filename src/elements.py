"""
elements.py
Author: bedlamzd of MT.lab

Классы для переопределения элементов в dxf для удобства использования,
т.к. ezdxf не предоставляет методов необходимых для решения задачи.
"""

from typing import List, Union, Optional, Tuple, Dict
from itertools import count
from ezdxf.math.vector import Vector, NULLVEC
from ezdxf.math.bspline import BSpline
from re import findall
import numpy as np
import utilities
from utilities import X, Y, Z, pairwise, apprx_point_height, triangle_area
from numpy import cos, sin, pi


# TODO: нарзека одновременно с расчётом Z координаты


class Element():
    """
    Общий класс с функциями общими для всех элементов, многие оверрайдятся в конкретных случаях
    """

    def __init__(self, entity, points: List['Vector'] = None):
        """
        Конструктор объекта

        :param entity: элемент из dxf
        """
        self.entity = entity
        self.points = points  # type: List[Vector]
        self.sliced = False
        self.with_z = False
        self.backwards = False
        self._length = None
        self._flat_length = None

    @property
    def first(self) -> Vector:
        return self.points[0] if not self.backwards else self.points[-1]

    @property
    def last(self) -> Vector:
        return self.points[-1] if not self.backwards else self.points[0]

    @property
    def centroid(self) -> Vector:
        try:
            return self._centroid
        except AttributeError:
            centroid = NULLVEC
            for p1, p2 in pairwise(self.points):
                centroid += p1.lerp(p2)
            self._centroid = centroid
            return centroid

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for v1, v2 in pairwise(self.points):
                length += v1.distance(v2)
            self._length = length
        return self._length

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for v1, v2 in pairwise(self.points):
                flat_length += v1.vec2.distance(v2.vec2)
            self._flat_length = flat_length
        return self._flat_length

    def __str__(self) -> str:
        return f'Element: {self.entity.dxftype()}\n ' + \
               f'first point: {self.first}\n ' + \
               f'last point: {self.last}'

    def __repr__(self) -> str:
        return f'Element: {self.entity.dxftype()}\n ' + \
               f'first point: {self.first}\n ' + \
               f'last point: {self.last}'

    def translate(self, vector: 'Vector' = NULLVEC):
        """
        Задать смещение для рисунка (добавить к нарезанным координатам смещение)

        :param vector: величина смещение
        :return: None
        """
        self.points = [v + vector for v in self.points]

    def rotate(self, angle: float, center: Vector = None):
        if center is not None:
            self.translate(-center)
        self.points = [v.rotate(angle) for v in self.points]
        if center is not None:
            self.translate(center)

    def best_distance(self, point: 'Vector' = NULLVEC) -> float:
        """
        Вычисляет с какой стороны точка находится ближе к элементу и ориентирует его соответственно

        :param point: точка от которой считается расстояние
        :return: минимальное расстояние до одного из концов объекта
        """
        dist2first = self.points[0].distance(point)
        dist2last = self.points[-1].distance(point)
        self.backwards = dist2last < dist2first
        return min(dist2first, dist2last)

    def get_points(self) -> List[Vector]:
        """
        Возвращает точки
        """
        return self.points if not self.backwards else self.points[::-1]

    def get_sliced_points(self) -> List[Vector]:
        """
        Возвращает нарезанные координаты
        """
        if self.sliced:
            return self.points if not self.backwards else self.points[::-1]
        else:
            return None

    def slice(self, step=1):
        """
        Нарезать элемент на более менее линии с заданным шагом
        :param float step: шаг нарезки
        :return:
        """
        sliced = [self.points[0]]
        for start, end in pairwise(self.points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
                sliced.append(end)
        self.points = sliced
        self.sliced = True
        self._length = None

    def add_z(self, height_map: Optional[np.ndarray] = None, point_apprx=False, **kwargs):
        if height_map is None:
            pass
        self.points = [v.replace(z=apprx_point_height(v, height_map, point_apprx=point_apprx, **kwargs)) for v in
                       self.points]
        self.with_z = True
        self._length = None


class Point(Element):
    # TODO: написать обработку точек
    pass


class Polyline(Element):
    """
    Подкласс для элемента Полилиния из dxf
    """

    def __init__(self, polyline):
        points = [Vector(point) for point in polyline.points()]
        super().__init__(polyline, points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            points = [Vector(point) for point in self.entity.points()]
            centroid = NULLVEC
            for p1, p2 in pairwise(points):
                centroid += p1.lerp(p2)
            self._centroid = centroid
            return centroid

    def slice(self, step=1):
        points = [Vector(point) for point in self.entity.points()]
        sliced = [points[0]]
        for start, end in pairwise(points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
                sliced.append(end)
        self.points = sliced
        self.sliced = True
        self._length = None


class LWPolyline(Polyline):
    # TODO: написать обработку LW полилиний
    pass


class Spline(Element, BSpline):
    """
    Подкласс для объека Сплайн
    """

    def __init__(self, spline):
        control_points = [Vector(point) for point in spline.control_points]
        knots = [knot for knot in spline.knots]
        weights = [weight for weight in spline.weights] if spline.weights else None
        order = spline.dxf.degree + 1
        BSpline.__init__(self, control_points, order, knots, weights)
        points = [point for point in self.approximate()]
        Element.__init__(self, spline, points)

    @property
    def first(self):
        return self.points[0] if not self.backwards else self.points[-1]

    @property
    def last(self):
        return self.points[-1] if not self.backwards else self.points[0]

    def slice(self, step=1):
        # TODO: подумать как использовать градиентный спуск или т.п.
        self.sliced = True
        points = [Vector(point) for point in self.approximate(int(self.max_t / step))]
        self.points = points
        self._length = None


class Line(Element):
    """
    Подкласс для объекта Линия
    """

    def __init__(self, line):
        points = [Vector(line.dxf.start), Vector(line.dxf.end)]
        super().__init__(line, points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            self._centroid = Vector(self.entity.dxf.start).lerp(self.entity.dxf.end)
            return self._centroid

    def slice(self, step=1):
        points = [Vector(self.entity.dxf.start), Vector(self.entity.dxf.end)]
        sliced = [points[0]]
        for start, end in pairwise(points):
            dist = start.distance(end)
            n_steps = int(dist / step)
            try:
                param_step = step / dist
            except ZeroDivisionError:
                continue
            v = Vector()
            for i in range(n_steps):
                v = start.lerp(end, param_step * (i + 1))
                sliced.append(v)
            if not v.isclose(end):
                sliced.append(end)
        self.points = sliced
        self.sliced = True
        self._length = None


class Circle(Element):
    """
    Подкласс для объекта Окружность
    """

    def __init__(self, circle):
        self.center = circle.dxf.center  # type: Vector
        self.radius = circle.dxf.radius  # type: float
        points = [self.center.replace(x=self.center.x + self.radius),
                  self.center.replace(x=self.center.x + self.radius)]
        super().__init__(circle, points=points)

    @property
    def flat_length(self):
        if self._flat_length is None:
            flat_length = 2 * pi * self.radius
            self._flat_length = flat_length
        return self._flat_length

    def slice(self, step=1):
        n_steps = int(self.flat_length / step)
        angle_step = 2 * pi / n_steps
        sliced = []
        v = Vector()
        for i in range(n_steps + 1):
            v = self.first - self.center
            v = v.rotate(i * angle_step)
            v += self.center
            sliced.append(v)
        if not v.isclose(self.last):
            sliced.append(self.last)
        self.points = sliced
        self.sliced = True
        self._length = None

    @property
    def centroid(self):
        return self.center


class Arc(Element):
    """
    Подклас для объекта Дуга
    """

    def __init__(self, arc):
        self.center = arc.dxf.center  # type: Vector
        self.radius = arc.dxf.radius  # type: float
        self.start_angle = arc.dxf.start_angle * pi / 180  # в радианах
        self.end_angle = arc.dxf.end_angle * pi / 180  # в радианах
        if self.start_angle > self.end_angle:
            self.end_angle += 2 * pi
        points = [Vector.from_angle(self.start_angle, self.radius) + self.center,
                  Vector.from_angle(self.end_angle, self.radius) + self.center]
        super().__init__(arc, points=points)

    @property
    def centroid(self):
        try:
            return self._centroid
        except AttributeError:
            centroid_x = self.radius / self.flat_length * (sin(self.end_angle) - sin(self.start_angle)) + self.center.x
            centroid_y = self.radius / self.flat_length * (cos(self.start_angle) - cos(self.end_angle)) + self.center.y
            self._centroid = Vector(centroid_x, centroid_y, 0)
            return self._centroid

    @property
    def flat_length(self):
        if self._flat_length is None:
            flat_length = (self.end_angle - self.start_angle) * self.radius
            self._flat_length = flat_length
        return self._flat_length

    def slice(self, step=1):
        n_steps = int(self.flat_length / step)
        angle_step = (self.end_angle - self.start_angle) / n_steps
        sliced = []
        v = Vector()
        for i in range(n_steps + 1):
            v = self.points[0] - self.center
            v = v.rotate(i * angle_step)
            v += self.center
            sliced.append(v)
        if not v.isclose(self.points[-1]):
            sliced.append(self.points[-1])
        self.sliced = True
        self.points = sliced
        self._length = None

    def __str__(self):
        return 'Arc object: ' + super().__str__()


class Ellipse(Element):
    # TODO: написать обработку эллипсов
    pass


class Contour:
    def __init__(self, elements: Union[List[Element], Element] = None):
        """
        :param elements: элементы составляющие контур
        """
        self._length = None
        self._flat_length = None
        if elements is None:
            self.elements = []
            self.closed = False
        else:
            if isinstance(elements, List):
                self.elements = elements
            elif isinstance(elements, Element):
                self.elements = [elements]
            else:
                raise TypeError('Contour should be either List[Element] or Element.')
            if self.first_point == self.last_point:
                self.closed = True
            else:
                self.closed = False

    def __add__(self, other: Union['Contour', Element]) -> 'Contour':
        if isinstance(other, Contour):
            if not len(self):
                elements = other.elements
                self._length = None
                self._flat_length = None
                return Contour(elements)
            if self.closed or other.closed:
                raise Exception('Cannot add closed contours.')
            """
             1. end to start
              c1 + c2
            2. end to end
              c1 + c2.reversed
            3. start to end
              c2 + c1
            4. start to start
              c2.reversed + c1
            """
            if self.last_point == other.first_point:
                elements = self.elements + other.elements
                # return Contour(elements)
            elif self.last_point == other.last_point:
                elements = self.elements + other.elements[::-1]
                # return Contour(elements)
            elif self.first_point == other.last_point:
                elements = other.elements + self.elements
                # return Contour(elements)
            elif self.first_point == other.first_point:
                elements = other.elements[::-1] + self.elements
                # return Contour(elements)
            else:
                raise Exception('Contours not connected.')
            # return Contour(elements)
        elif isinstance(other, Element):
            if self.last_point == other.first:
                elements = self.elements + [other]
                # return Contour(elements)
            elif self.last_point == other.last:
                elements = self.elements + [other]
                other.backwards = not other.backwards
                # return Contour(elements)
            elif self.first_point == other.last:
                elements = [other] + self.elements
                # return Contour(elements)
            elif self.first_point == other.first:
                elements = [other] + self.elements
                other.backwards = not other.backwards
                # return Contour(elements)
            else:
                raise Exception('Shapes not connected.')
        else:
            raise TypeError('Can add only Contour or Element')
        self._length = None
        self._flat_length = None
        return Contour(elements)

    def add_element(self, element: Element):
        if element in self.elements:
            raise Exception('Element is in contour already.')
        if not isinstance(element, Element):
            raise TypeError('Adding object should be Element.')
        if self.first_point == element.last:
            self.elements = [element] + self.elements
        elif self.last_point == element.first:
            self.elements += [element]
        elif self.first_point == element.first:
            element.backwards = not element.backwards
            self.elements = [element] + self.elements
        elif self.last_point == element.last:
            element.backwards = not element.backwards
            self.elements += [element]
        else:
            raise Exception('Element does not connected to contour.')

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item: Union[int, slice]) -> Element:
        return self.elements[item]

    def __reversed__(self):
        for element in self.elements[::-1]:
            yield element

    @property
    def first_element(self) -> Element:
        return self.elements[0]

    @property
    def last_element(self) -> Element:
        return self.elements[-1]

    @property
    def first_point(self) -> Vector:
        return self.first_element.first

    @property
    def last_point(self) -> Vector:
        return self.last_element.last

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for element in self.elements:
                flat_length += element.flat_length
            self._flat_length = flat_length
        return self._flat_length

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
        return self._length

    def isclose(self, other: Union[Vector, Element, "Contour"], abs_tol: float = 1e-12) -> bool:
        if isinstance(other, Vector):
            close2first = self.first_point.isclose(other, abs_tol)
            close2last = self.last_point.isclose(other, abs_tol)
            return close2first or close2last
        elif isinstance(other, Element):
            close2first = self.first_point.isclose(other.first, abs_tol) or self.first_point.isclose(other.last,
                                                                                                     abs_tol)
            close2last = self.last_point.isclose(other.first, abs_tol) or self.last_point.isclose(other.last, abs_tol)
            return close2first or close2last
        elif isinstance(other, Contour):
            close2first = self.first_point.isclose(other.first_point, abs_tol) or self.first_point.isclose(
                other.last_point,
                abs_tol)
            close2last = self.last_point.isclose(other.first_point, abs_tol) or self.last_point.isclose(
                other.last_point,
                abs_tol)
            return close2first or close2last
        else:
            raise TypeError('Should be Vector or Element or Contour.')

    def best_distance(self, point: Vector = NULLVEC) -> float:
        dist2first = 0 if self.first_point == point else self.first_point.distance(point)
        dist2last = 0 if self.last_point == point else self.last_point.distance(point)
        return min(dist2first, dist2last)

    def get_points(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.get_points()
        return points

    def get_sliced_points(self) -> List[Vector]:
        points = []
        for element in self.elements:
            points += element.get_sliced_points()
        return points


class Layer:
    number_generator = count()

    def __init__(self, name=None, contours: Union[List[Contour], Contour] = None, priority=None):
        if isinstance(contours, List):
            self.contours = contours
        elif isinstance(contours, Contour):
            self.contours = [contours]
        elif contours is None:
            self.contours = []
        self.number = next(Layer.number_generator)
        self.name = name if name is not None else f'Layer {self.number}'
        self.cookieContour = True if name == 'Contour' else False
        self.priority = priority if priority is not None else 0

    def add_contour(self, contours: Union[List[Contour], Contour]):
        if isinstance(contours, List):
            self.contours += contours
        elif isinstance(contours, Contour):
            self.contours += [contours]

    def get_elements(self):
        elements = []
        for contour in self.contours:
            elements += contour.elements
        return elements


class Drawing:
    """

    Attributes:
        dxf: An ezdxf Drawing which basically contains all the necessary data
        modelspace: A dxf.modelspace(), only for a convenience
        layers Dict[str, Layer] : A dict of [layer.name, layer]
        elements List[Element]: Contains all graphic entities from dxf.
        contours List[Contour]: Contains all contours found in layers.
        center Vector: Drawing geometrical center.
        rotation float: Drawing angle or orientation.
        organized bool: True if elements are ordered and contours are constructed
    """

    def __init__(self, dxf=None, center: Vector = None, rotation: float = None):
        """
        :param dxf: открытый библиотекой рисунок
        :param center: смещение центра рисунка
        :param rotation: угол поворота рисунка (его ориентация)
        lookup Drawing for more
        """
        self.layers = {}  # type: Dict[str, Layer]
        self.elements = []  # type: List[Element]
        self.contours = []  # type: List[Contour]
        self.organized = False  # type: bool
        self._length = None
        self._flat_length = None
        if dxf is None:
            self.dxf = None
            self.modelspace = None
        else:
            self.dxf = dxf
            self.modelspace = self.dxf.modelspace()
            self.read_by_layer()
        self._center, self._rotation = self.find_center_and_rotation()
        if center is not None:
            self.center = center
        if rotation is not None:
            self.rotation = rotation

    def __str__(self):
        return f'Геометрический центр рисунка: X: {self.center[X]:4.2f} Y: {self.center[Y]:4.2f} мм\n' + \
               f'Ориентация рисунка: {self.rotation * 180 / pi: 4.2f} градуса\n' + \
               f'Общая плоская длина рисунка: {self.flat_length: 4.2f} мм'

    @property
    def center(self) -> Vector:
        return self._center

    @center.setter
    def center(self, center: Union[Vector, List[float], Tuple[float]]):
        center = Vector(center)
        self.translate(center - self._center)
        self._center = center

    def translate(self, vector: Vector):
        for element in self.elements:
            element.translate(vector)

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, angle: float):
        self.rotate(angle - self._rotation)
        self._rotation = angle

    def rotate(self, angle: float):
        for element in self.elements:
            element.rotate(angle, self.center)

    def find_center_and_rotation(self) -> Tuple[Vector, float]:
        """
        Расчитывает геометрический центр рисунка
        :return:
        """
        cookie_contour_layer = self.layers.get('Contour')
        if cookie_contour_layer is None:
            # TODO: place warning here
            return NULLVEC, 0
        else:
            points = []
            for element in cookie_contour_layer.get_elements():
                element.slice(0.1)
                points += element.get_points()
            points = np.asarray([list(v.vec2) for v in points], dtype=np.float32)
            center, rotation = utilities.find_center_and_rotation(points[:, np.newaxis, :], True)
            return center, rotation

    @property
    def length(self) -> float:
        if self._length is None:
            length = 0
            for element in self.elements:
                length += element.length
            self._length = length
        return self._length

    @property
    def flat_length(self) -> float:
        if self._flat_length is None:
            flat_length = 0
            for element in self.elements:
                flat_length += element.flat_length
            self._flat_length = flat_length
        return self._flat_length

    def read_dxf(self, root):
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                self.read_dxf(block)
            elif element_redef(element):
                self.elements.append(element_redef(element))
        self.organized = False
        print('dxf прочтён.')

    def read_entities(self, root, entities=None):
        if entities is None:
            entities = []
        for element in root:
            if element.dxftype() == 'INSERT':
                block = self.dxf.blocks[element.dxf.name]
                entities += self.read_entities(block)
            elif element_redef(element):
                entities.append(element_redef(element))
        print('элементы получены')
        return entities

    def read_by_layer(self):
        layers = {}
        elements_in_dwg = []
        contours_in_dwg = []
        for layer in self.dxf.layers:
            name = layer.dxf.name
            print(f'чтение слоя {name}')
            if name == 'Defpoints':
                print('    пропуск')
                continue
            priority = findall('\d+', name)
            priority = int(priority[0]) if priority else None
            entities_in_layer = self.modelspace.query(f'*[layer=="{name}"]')
            entities_in_layer = self.read_entities(entities_in_layer)
            if not entities_in_layer:
                continue
            entities_in_layer = self.organize_entities(entities_in_layer)
            elements_in_dwg += entities_in_layer
            contours_in_layer = self.make_contours(entities_in_layer)
            contours_in_dwg += contours_in_layer
            layers[name] = Layer(name, contours_in_layer, priority)
        self.layers = layers
        self.elements = elements_in_dwg
        self.contours = contours_in_dwg
        self.organized = True
        print('файл прочтён')

    def slice(self, step: float = 1.0):
        for element in self.elements:
            element.slice(step)
        self._length = None
        print(f'Объекты нарезаны с шагом {step:2.1f} мм')

    def add_z(self, height_map: Optional[np.ndarray] = None, point_apprx=False, **kwargs):
        for element in self.elements:
            element.add_z(height_map, point_apprx=point_apprx, **kwargs)
        self._length = None

    def organize_entities(self, entities: List[Element], start_point: Vector = NULLVEC):
        path = []
        elements = entities
        # сортировать элементы по их удалению от точки
        elements.sort(key=lambda x: x.best_distance(start_point))
        while len(elements) != 0:
            # первый элемент в списке (ближайший к заданной точке) - текущий
            current = elements[0]
            # добавить его в сориентированный массив
            path.append(current)
            # убрать этот элемент из неотсортированного списка
            elements.pop(0)
            # отсортировать элементы по их удалению от последней точки предыдущего элемента
            elements.sort(key=lambda x: x.best_distance(current.last))
        print('элементы отсортированы')
        return path

    def make_contours(self, entities: List[Element]):
        contour = Contour([entities[0]])
        contours = []
        for element in entities[1:]:
            if contour.isclose(element) and not contour.closed:
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        i = -1
        while i < len(contours) - 1:
            if contours[i].isclose(contours[i + 1]) and not contours[i].closed and not contours[i + 1].closed:
                if i == -1:
                    contours[i + 1] = contours[i] + contours[i + 1]
                    del contours[i]
                else:
                    contours[i:i + 2] = [contours[i] + contours[i + 1]]
            else:
                i += 1
        print('контуры составлены')
        return contours

    def organize_elements(self, start_point=(0, 0)):
        """
        Сортирует и ориентирует элементы друг за другом относительно данной точки
        :param start_point: точка, относительно которой выбирается первый элемент
        :return list of Element path: отсортированный и ориентированный массив элементов
        """
        path = []
        elements = self.elements.copy()
        # сортировать элементы по их удалению от точки
        elements.sort(key=lambda x: x.best_distance(start_point))
        while len(elements) != 0:
            # первый элемент в списке (ближайший к заданной точке) - текущий
            current = elements[0]
            # добавить его в сориентированный массив
            path.append(current)
            # убрать этот элемент из неотсортированного списка
            elements.pop(0)
            # отсортировать элементы по их удалению от последней точки предыдущего элемента
            elements.sort(key=lambda x: x.best_distance(current.last))
        self.elements = path
        self.organized = True
        print('Сформирована очередность элементов.')

    def find_contours(self):
        contour = Contour([self.elements[0]])
        contours = []
        for element in self.elements[1:]:
            if contour.isclose(element):
                contour += element
            else:
                contours.append(contour)
                contour = Contour([element])
        contours.append(contour)
        i = -1
        while i < len(contours) - 1:
            if contours[i].isclose(contours[i + 1]):
                if i == -1:
                    contours[i + 1] = contours[i] + contours[i + 1]
                    del contours[i]
                else:
                    contours[i:i + 2] = [contours[i] + contours[i + 1]]
            else:
                i += 1
        self.contours = contours
        print('Найдены контуры.')


def get_centroid(poly):
    """Calculates the centroid of a non-intersecting polygon.
    Args:
        poly: a list of points, each of which is a list of the form [x, y].
    Returns:
        the centroid of the polygon in the form [x, y].
    Raises:
        ValueError: if poly has less than 3 points or the points are not
                    formatted correctly.
    """
    # Make sure poly is formatted correctly
    if len(poly) < 3:
        raise ValueError('polygon has less than 3 points')
    for point in poly:
        if 2 != len(point):
            raise ValueError('point is not a list of length 2')
    # Calculate the centroid from the weighted average of the polygon's
    # constituent triangles
    area_total = 0
    centroid_total = [float(poly[0][0]), float(poly[0][1])]
    for i in range(0, len(poly) - 2):
        # Get points for triangle ABC
        a, b, c = poly[0], poly[i + 1], poly[i + 2]
        # Calculate the signed area of triangle ABC
        area = triangle_area(a, b, c, True)
        # If the area is zero, the triangle's line segments are
        # colinear so we should skip it
        if 0 == area:
            continue
        # The centroid of the triangle ABC is the average of its three
        # vertices
        centroid = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]
        # Add triangle ABC's area and centroid to the weighted average
        centroid_total[0] = ((area_total * centroid_total[0]) +
                             (area * centroid[0])) / (area_total + area)
        centroid_total[1] = ((area_total * centroid_total[1]) +
                             (area * centroid[1])) / (area_total + area)
        area_total += area
    return centroid_total


def element_redef(element) -> Optional[Element]:
    """
    Функция для переопределения полученного элемента в соответствующий подкласс класса Element

    :param element: элемент из dxf
    :return: переопределение этого элемента
    """
    if element.dxftype() == 'POLYLINE':
        return Polyline(element)
    elif element.dxftype() == 'SPLINE':
        return Spline(element)
    elif element.dxftype() == 'LINE':
        return Line(element)
    elif element.dxftype() == 'CIRCLE':
        return Circle(element)
    elif element.dxftype() == 'ARC':
        return Arc(element)
    elif element.dxftype() == 'ELLIPSE':
        pass
    elif element.dxftype() == 'LWPOLYLINE':
        pass
    elif element.dxftype() == 'POINT':
        pass
    else:
        print('Unknown element.')
        return None
