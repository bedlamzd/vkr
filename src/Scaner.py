import cv2
import numpy as np
from numpy import tan
from typing import Tuple
import Camera, Laser
import utilities


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


# TODO: Методы получения лазера отдельными функциями

class Scaner:
    def __init__(self, camera: Camera, laser: Laser, distance: float, angle: float):
        self.h = distance / self.tg_angle
        self.tg_angle = tan(angle)
        self.extraction_mode = 'max_peak'
        self.d = distance
        self.angle = angle
        self.velocity = 0  # mm/s
        self.camera = camera
        self.laser = laser
        self.cloud_shape = (self.camera.cap.get(cv2.CAP_PROP_FRAME_COUNT) // 1, self.camera.frame_width)
        self.cloud = np.zeros((*self.cloud_shape, 3))

    @property
    def depthmap(self):
        return utilities.normalize(self.cloud[..., -1])

    def find_local_coords(self, laser: np.ndarray):
        dy = laser - self.camera.v0
        dx = np.mgrid[laser.size] - self.camera.u0
        h = self.h * dy / (dy + self.camera.f * self.tg_angle)
        x = (self.h - h) * dx / self.camera.f
        y = h * self.tg_angle
        z = self.h - h
        return np.column_stack([x, y, z])

    def local2global_coords(self, local_coords: np.ndarray):
        global_coords = (self.camera.rot_mtx @ local_coords.T + self.camera.tvec).T
        return global_coords

    def extract_laser(self, img) -> np.ndarray:
        def max_peak(img) -> np.ndarray:
            img = self.apply_mask(self.get_blur(img), self.get_mask(img))

        def log() -> np.ndarray:
            pass

        def ggm() -> np.ndarray:
            pass

        modes = (max_peak, log, ggm)
        for mode in modes:
            if mode.__name__ == self.extraction_mode:
                return mode(img)

    def scan(self):
        camera = self.camera
        ret, img = camera.read_proc()
        while ret:
            laser = self.extract_laser(img)
            local_coords = self.find_local_coords(laser)
            global_coords = self.local2global_coords(local_coords)
            self.cloud[camera.current_frame_idx] = global_coords
            camera.tvec[0] += camera.current_frame_idx / camera.fps * self.velocity  # using FPS
            # camera.tvec[0] += camera.frame_timing*self.velocity # using timing
            ret, img = camera.read_proc()
