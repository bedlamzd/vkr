import cv2
import numpy as np
from numpy import tan
from typing import Tuple, Optional
from Camera import Camera
import utilities


# TODO: Методы получения лазера отдельными функциями
# TODO: logging
# TODO: World Frame binding via markers


class Scaner:
    def __init__(self, camera: Camera,
                 height: Optional[float] = None,
                 angle: Optional[float] = None,
                 velocity: Optional[float] = None,
                 img_proc_opts: Optional[dict] = None,
                 extraction_mode: str = 'max_peak',
                 extraction_opts: Optional[dict] = None):
        camera, height, angle, velocity, img_proc_opts, extraction_opts = self.check_parameters(camera=camera,
                                                                                                height=height,
                                                                                                angle=angle,
                                                                                                velocity=velocity,
                                                                                                img_proc_opts=img_proc_opts,
                                                                                                extraction_opts=extraction_opts)
        # config values
        self.h = height  # config
        self.tg_angle = tan(angle)  # config
        self.extraction_mode = extraction_mode  # config
        self.velocity = velocity  # mm/s, config
        self.camera = camera
        self.img_proc_opts = img_proc_opts
        self.extraction_opts = extraction_opts
        # calculated values
        self.d = height * self.tg_angle
        self.angle = angle
        self._cloud = np.zeros((self.camera.frame_count, self.camera.frame_width, 3))

    @staticmethod
    def check_parameters(camera=None, height=None, angle=None, velocity=None, img_proc_opts=None, extraction_opts=None):
        # TODO: write parameters validation
        if img_proc_opts is None:
            img_proc_opts = {'mask': True, 'color_filt': True, 'roi': True}
        else:
            img_proc_opts = img_proc_opts
        return camera, height, angle, velocity, img_proc_opts, extraction_opts

    @classmethod
    def from_json(cls, camera: Camera, filepath: str) -> 'Scaner':
        # TODO: write json parsing
        h, angle, extraction_mode, velocity = (filepath)
        return Scaner(camera, h, angle, velocity)

    @classmethod
    def from_config(cls, camera: Camera, filepath: str) -> 'Scaner':
        # TODO: write ini parsing
        h, angle, extraction_mode, velocity = (filepath)
        return Scaner(camera, h, angle, velocity)

    @staticmethod
    def refine_laser_center(prev=(0, 0), middle=(0, 0), nxt=(0, 0)) -> Tuple[float, float]:
        """
        Аппроксимирует по трём точкам параболу и находит её вершину
        Таким образом более точно находит позицию лазера в изображении

        :param Tuple[int, float] prev: предыдущая точка от m (m-1), (ряд, интенсивность)
        :param Tuple[int, float] middle: точка с максимальной интенсивностью, (ряд, интенсивность)
        :param Tuple[int, float] nxt: следующая точка от m (m+1), (ряд, интенсивность)
        :return: уточнённая позиция лазера с субпиксельной точностью и её аппроксимированная интенсивность

        a, b, c - параметры квадратичной функции
        y = ax^2 + bx + c
        """
        if prev[0] == middle[0] == nxt[0]:  # если точки совпадают, аппроксимация не получится, вернуть среднюю
            return middle
        a = .5 * (nxt[1] + prev[1]) - middle[1]
        if a == 0:  # если а = 0, то получилась линия, вершины нет, вернуть среднюю точку
            return middle
        b = (middle[1] - prev[1]) - a * (2 * middle[0] - 1)
        c = prev[1] - prev[0] * (a * prev[0] + b)
        row = -b / (2 * a)
        intensity = a * row ** 2 + b * row + c
        return row, intensity

    @staticmethod
    def rough_laser(img: np.ndarray) -> np.ndarray:
        return np.argmax(img, axis=0)

    @classmethod
    def fine_laser(cls, img: np.ndarray) -> np.ndarray:
        laser = cls.rough_laser(img)
        for col, row in enumerate(laser):
            if row == 0 or row == img.shape[0] - 1:
                continue
            prev_row, next_row = row - 1, row + 1
            prev = prev_row, img[prev_row, col]
            middle = row, img[row, col]
            nxt = next_row, img[next_row, col]
            laser[col], _ = cls.refine_laser_center(prev, middle, nxt)
        return laser

    @classmethod
    def max_peak(cls, img: np.ndarray) -> np.ndarray:
        return cls.fine_laser(img)

    def laplace_of_gauss(self, img: np.ndarray) -> np.ndarray:
        ksize, sigma = self.extraction_opts.get('ksize', 3), self.extraction_opts.get('sigma', 0)
        kernel_x = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
        kernel_y = kernel_x.T
        gauss = -cv2.sepFilter2D(img, cv2.CV_64F, kernel_x, kernel_y)
        log_img = cv2.Laplacian(gauss, cv2.CV_64F)
        log_img = self.camera.apply_mask(log_img, self.camera.get_mask(self.camera.apply_blur(img)))
        log_img[log_img < 0] = 0
        laser = self.fine_laser(log_img)
        return laser

    @staticmethod
    def ggm(img: np.ndarray) -> np.ndarray:
        ggm = img.astype(np.float32) / np.amax(img)
        laser = np.sum(ggm * (np.mgrid[:ggm.shape[0], :ggm.shape[1]][0] + 1), axis=0) / np.sum(ggm, axis=0) - 1
        laser[np.isinf(laser) | np.isnan(laser)] = 0
        return laser

    @staticmethod
    def iggm(img) -> np.ndarray:
        pass

    @property
    def depthmap(self) -> np.ndarray:
        return utilities.normalize(self._cloud[..., -1]).copy()

    @property
    def pointcloud(self) -> np.ndarray:
        return self._cloud.copy()

    def find_local_coords(self, laser: np.ndarray) -> np.ndarray:
        laser = laser + self.camera.roi[1]
        laser = np.pad(laser, (self.camera.roi[0], self.camera.frame_width - (self.camera.roi[0] + self.camera.roi[2])))
        dy = laser - self.camera.v0
        dx = np.mgrid[:laser.size] - self.camera.u0
        tg_beta = dy / self.camera.fy
        tg_gamma = dx / self.camera.fx
        z = self.h * self.tg_angle / (self.tg_angle + tg_beta)
        y = z * tg_beta
        x = z * tg_gamma
        return np.column_stack([x, y, z])

    def local2global_coords(self, local_coords: np.ndarray) -> np.ndarray:
        global_coords = ((self.camera.rot_mtx @ local_coords.T).T + self.camera.tvec)
        return global_coords

    def scan(self):
        camera = self.camera
        ret, img = camera.read_processed(**self.img_proc_opts)
        while ret:
            laser = getattr(self, self.extraction_mode)(img)
            local_coords = self.find_local_coords(laser)
            global_coords = self.local2global_coords(local_coords)
            self._cloud[camera.current_frame_idx] = global_coords
            camera.tvec[0] += self.velocity / camera.fps  # using FPS
            # camera.tvec[0] += camera.frame_timing * self.velocity - camera.tvec[0] # using timing
            ret, img = camera.read_processed()


def scan(videopath, cameraconfig, scanerconfig):
    cap = cv2.VideoCapture(videopath)
    camera = Camera.Camera.from_json_file(cameraconfig, cap)
    scaner = Scaner.from_json_file(camera, scanerconfig)
    scaner.scan()
    depthmap = scaner.depthmap
    pointcloud = scaner.pointcloud
