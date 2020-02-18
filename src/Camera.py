import numpy as np
from numpy import cos, sin, tan, pi
from typing import Optional
from imutils import rotate_bound
import cv2
from cv2 import VideoCapture, rotate, ROTATE_180
from typing import Tuple, List


def find_laser_center(p=(0, 0), m=(0, 0), n=(0, 0)) -> Tuple[float, float]:
    """
    Аппроксимирует по трём точкам параболу и находит её вершину
    Таким образом более точно находит позицию лазера в изображении

    :param Tuple[int, float] p: предыдущая точка от m (m-1)
    :param Tuple[int, float] m: точка с максимальной интенсивностью, (ряд, интенсивность)
    :param Tuple[int, float] n: следующая точка от m (m+1)
    :return: уточнённая позиция лазера с субпиксельной точностью и её аппроксимированная интенсивность

    a, b, c - параметры квадратичной функции
    y = ax^2 + bx + c
    """
    if p[0] == m[0] or m[0] == n[0]:  # если точки совпадают, аппроксимация не получится, вернуть среднюю
        return m
    a = .5 * (n[1] + p[1]) - m[1]
    if a == 0:  # если а = 0, то получилась линия, вершины нет, вернуть среднюю точку
        return m
    b = (m[1] - p[1]) - a * (2 * m[0] - 1)
    c = p[1] - p[0] * (a * p[0] + b)
    xc = -b / (2 * a)
    yc = a * xc ** 2 + b * xc + c
    return xc, yc


def predict_laser(img: np.ndarray, row_start=0, row_stop=None) -> np.ndarray:
    # TODO: написать варианты не использующие LoG:
    #       3. применять IGGM (возможно замедление работы алгоритма)
    """

    :param img: preprocessed img
    :param row_start: минимально возможный ряд
    :param row_stop: максимально возможный ряд
    :return fine_laser: list of predicted laser subpixel positions
    """
    laser = np.argmax(img, axis=0)
    laser[laser > (row_stop - row_start - 1)] = 0
    fine_laser = np.zeros(laser.shape)
    for column, row in enumerate(laser):
        if row == 0:
            continue
        prevRow = row - 1
        nextRow = row + 1 if row < img.shape[0] - 1 else img.shape[0] - 1
        p1 = (1. * prevRow, 1. * img[prevRow, column])
        p2 = (1. * row, 1. * img[row, column])
        p3 = (1. * nextRow, 1. * img[nextRow, column])
        fine_laser[column] = find_laser_center(p1, p2, p3)[0] + row_start
    fine_laser[fine_laser > row_stop - 1] = row_stop - 1
    return fine_laser


class Camera:
    def __init__(self, *, focal_length=None, pixel_size=1, frame_size=(480, 640), principal_point=None, mtx=None):
        self.f = focal_length  # фокусное расстояние
        self.pixel_size = pixel_size  # размер пикселя
        self.frame_size = frame_size
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]
        self.u0 = self.frame_width // 2 - 1
        self.v0 = self.frame_height // 2 - 1
        self.matrix = mtx or np.array([[self.f / self.pixel_size, 0, self.frame_width // 2 - 1],
                                       [0, self.f / self.frame_height],
                                       [0, 0, 1]])
        self.rot_mtx = np.eye(3)
        self.tvec = np.array([0, 0, 0])
        self._cap = None  # type: Optional[VideoCapture]
        self.roi = None  # ((x0, x1), (y0,y1))
        self.ksize = 29
        self.sigma = 4.45
        self.thresh = 0

    def read_params(self, file):
        pass

    def find_pose(self, img, grid, **kwargs):
        pass

    @property
    def cap(self) -> VideoCapture:
        assert isinstance(self._cap, VideoCapture)
        return self._cap  # type: VideoCapture

    @cap.setter
    def cap(self, cap: VideoCapture):
        assert isinstance(cap, VideoCapture)
        self._cap = cap

    def prepare_img(self, img: np.ndarray):
        new_img = img.copy()
        if self.roi:
            (x0, x1), (y0, y1) = self.roi
            new_img = new_img[x0:x1, y0:y1]
        cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY, new_img)
        return new_img

    def get_mask(self, img):
        if self.thresh == 0:
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.thresh > 0:
            _, mask = cv2.threshold(img, 0, self.thresh, cv2.THRESH_BINARY)
        else:
            mask = np.full_like(img, 255, np.uint8)
        return mask

    def get_blur(self, img):
        return cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma)

    def apply_mask(self, img, mask):
        return cv2.bitwise_and(img, img, mask=mask)

    def read(self):
        if self._cap:
            ret, img = self._cap.read()
            if ret:
                return ret, img
        else:
            raise Exception
