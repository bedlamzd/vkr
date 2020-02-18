import cv2
import numpy as np
from numpy import tan
import Camera, Laser
import utilities


# TODO: Методы получения лазера отдельными функциями

class Scaner:
    def __init__(self, camera: Camera, laser: Laser, distance: float, angle: float):
        self.extraction_mode = 'max_peak'
        self.d = distance
        self.angle = angle
        self.tg_angle = tan(angle)
        self.h = distance / self.tg_angle
        self.camera = camera
        self.laser = laser
        self.cloud_shape = (self.camera.cap.get(cv2.CAP_PROP_FRAME_COUNT) // 1, self.camera.frame_width)
        self.cloud = np.zeros((*self.cloud_shape, 3))

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

    def extract_laser(self, img):
        def max_peak(img):
            img = self.apply_mask(self.get_blur(img), self.get_mask(img))

        def log():
            pass

        def ggm():
            pass

        modes = (max_peak, log, ggm)
        for mode in modes:
            if mode.__name__ == self.extraction_mode:
                return mode(img)

    def process_img(self, img):
        pass

    @property
    def depthmap(self):
        return utilities.normalize(self.cloud[..., -1])
