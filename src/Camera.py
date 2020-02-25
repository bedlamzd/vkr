import numpy as np
from numpy import cos, sin, tan, pi
from typing import Optional
from imutils import rotate_bound
import cv2
from cv2 import VideoCapture, rotate, ROTATE_180
from typing import Tuple, List, Iterable
from utilities import Error
import configparser


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
    """
    A class representing a camera

    :ivar mtx: camera intrinsic matrix [[fx,  0, u0],
                                        [ 0, fy, v0],
                                        [ 0,  0,  1]]
    :type mtx: np.ndarray
    :ivar rot_mtx: rotation matrix from camera coordinate system (CCS) to global
    :type rot_mtx: np.ndarray
    :ivar tvec: translation vector, camera coordinates in GCS
    :type tvec: np.ndarray
    :ivar roi: region of interest in camera view ((x0,y0),(x1,y1))
    :type roi: Iterable
    :ivar ksize: kernel size for gaussian blur
    :type ksize: int
    :ivar sigma: sigma parameter for gaussian blur
    :type sigma: float
    :ivar threshold: threshold value
    :type threshold: float
    """

    def __init__(self):
        self.mtx = np.array([[self.fx, 0, self.u0],
                             [0, self.fy, self.v0],
                             [0, 0, 1]])
        self.rot_mtx = np.eye(3)
        self.tvec = np.array([0, 0, 0])
        self.roi = ((0, 0), (-1, -1))  # ((x0, y0), (x1,y1))
        self.ksize = 29
        self.sigma = 4.45
        self.threshold = 0
        self._cap = None  # type: Optional[VideoCapture]
        self._frame_size = 0
        self._frame_width = 0
        self._frame_height = 0

    def read_settings(self, file):
        from os.path import splitext
        _, ext = splitext(file)
        if ext == '.json':
            import json
            with open(file) as f:
                jsn = json.load(f)
            for key in vars(self):
                if key in jsn:
                    setattr(self, key, jsn[key])

        elif ext == '.ini':
            import configparser

    def find_pose(self, img, grid, **kwargs):
        pass

    @property
    def u0(self):
        return self.mtx[0, 2]

    @property
    def v0(self):
        return self.mtx[1, 2]

    @property
    def fx(self):
        return self.mtx[0, 0]

    @property
    def fy(self):
        return self.mtx[1, 1]

    @property
    def focal_length(self):
        return (self.fx, self.fy)

    @property
    def cap(self) -> VideoCapture:
        assert isinstance(self._cap, VideoCapture)
        return self._cap  # type: VideoCapture

    @cap.setter
    def cap(self, cap: VideoCapture):
        assert isinstance(cap, VideoCapture)
        self._cap = cap

    @property
    def current_frame_idx(self):
        return self.next_frame_idx - 1

    @property
    def next_frame_idx(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    @property
    def frame_width(self):
        # TODO: записывать эти параметры в _frame_* и возвращать их?
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def frame_size(self):
        return (self.frame_width, self.frame_height)

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def prepare_img(self, img: np.ndarray):
        new_img = img.copy()
        if self.roi:
            (x0, x1), (y0, y1) = self.roi
            new_img = new_img[x0:x1, y0:y1]
        cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY, new_img)
        return new_img

    def get_mask(self, img):
        if self.threshold == 0:
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold > 0:
            _, mask = cv2.threshold(img, 0, self.threshold, cv2.THRESH_BINARY)
        else:
            mask = np.full_like(img, 255, np.uint8)
        return mask

    def get_blur(self, img):
        return cv2.GaussianBlur(img, (self.ksize, self.ksize), self.sigma)

    def apply_mask(self, img, mask):
        return cv2.bitwise_and(img, img, mask=mask)

    def read_raw(self):
        if self._cap:
            ret, img = self._cap.read()
            if ret:
                return ret, img
        else:
            raise Error('cap is not set')

    def read_proc(self):
        img = self.read_raw()
