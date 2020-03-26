import cv2
import json
import numpy as np
from numpy import tan
from typing import Tuple, Optional, Sequence
from Camera import Camera
from tools.general import normalize, avg

from itertools import combinations


# TODO: logging
# TODO: World Frame binding via markers


class Scanner:
    _config_attr = [
        'height',
        'angle',
        'velocity',
        'img_proc_opts',
        'extraction_mode',
        'extraction_opts'
    ]

    def __init__(self, camera: Optional[Camera] = None,
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
        # TODO: Sequence -> numpy.ndarray conversion
        # TODO: extraction mode names formatting and aliasing (e.g. 'Laplace of Gauss'|'log' -> 'laplace_of_gauss')
        if img_proc_opts is None:
            img_proc_opts = {'mask': True, 'color_filt': True, 'roi': True}
        else:
            img_proc_opts = img_proc_opts
        return camera, height, angle, velocity, img_proc_opts, extraction_opts

    @property
    def config_data(self):
        return {attribute: getattr(self, attribute).tolist() if hasattr(getattr(self, attribute), 'tolist')
        else getattr(self, attribute) for attribute in self._config_attr}

    @classmethod
    def from_json(cls, camera: Camera, filepath: str) -> 'Scanner':
        data = json.load(open(filepath))
        data = {attr: np.array(value) if isinstance(value, Sequence) else value
                for attr, value in data.items() if attr in cls._config_attr}
        return cls(camera=camera, **data)

    def dump_json(self, filepath='src/scanner.json', **kwargs):
        json.dump(self.config_data, open(filepath, 'w'), **kwargs)
        print(f'Scanner data saved to {filepath}')

    def dumps_json(self, **kwargs):
        return json.dumps(self.config_data, **kwargs)

    @classmethod
    def from_config(cls, camera: Camera, filepath: str) -> 'Scanner':
        # TODO: write ini parsing
        h, angle, extraction_mode, velocity = (filepath)
        return cls(camera, h, angle, velocity)

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
        return normalize(self._cloud[..., -1]).copy()

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


class ScannerCalibrator:
    def __init__(self, scanner):
        self.scanner = scanner
        self.tg_alpha = 0
        self.h = 0

    @property
    def alpha(self):
        return np.arctan(self.tg_alpha)

    def calculate_from_angles(self,
                              h1, beta1,
                              h2, beta2,
                              beta0=0, r32=0, r33=1, *,
                              rot_mtx=None):
        tg_beta0, tg_beta1, tg_beta2 = np.tan([beta0, beta1, beta2])
        tg_alpha, H = self.calculate_from_tangent(h1, tg_beta1, h2, tg_beta2, tg_beta0, r32, r33, rot_mtx=rot_mtx)
        return tg_alpha, H

    def calculate_from_tangent(self,
                               h1, tg_beta1,
                               h2, tg_beta2,
                               tg_beta0=0, r32=0, r33=1, *,
                               rot_mtx=None):
        if rot_mtx:
            r32, r33 = rot_mtx[3, 2], rot_mtx[3, 2]
        tg_alpha = ((h1 * tg_beta1 * (tg_beta2 - tg_beta0) - h2 * tg_beta2 * (tg_beta1 - tg_beta0))
                    / (h1 * (tg_beta2 - tg_beta0) - h2 * (tg_beta1 - tg_beta0)))
        H = h1 * (((tg_alpha + tg_beta1) * (tg_alpha + tg_beta0))
                  / ((tg_beta1 - tg_beta0) * (r32 * tg_alpha ** 2 - r33 * tg_alpha)))
        return tg_alpha, H

    def calibrate_from_images(self, images, heights):
        u0, v0 = int(self.scanner.camera.u0), self.scanner.camera.v0  # optical center of an image
        results = []  # (tg_alpha, H) pairs for each calculation
        # tangent of the angle between laser ray and optical axis of the camera in each image
        tg_beta = [(self.scanner.laplace_of_gauss(image)[u0] - v0) / self.scanner.camera.fy for image in images]
        tg_beta0 = tg_beta[0]  # tangent for zero level
        for (tg_betai, hi), (tg_betak, hk) in combinations(zip(tg_beta[1:], heights[1:]), 2):
            results.append(self.calculate_from_tangent(hi, tg_betai,
                                                       hk, tg_betak,
                                                       tg_beta0, rot_mtx=self.scanner.camera.rot_mtx))
        self.tg_alpha, self.h = [avg(*result) for result in zip(*results)]  # find average of all calculations
        return self.tg_alpha, self.h


def scan(video_path: str, camera_config: str, scaner_config: str):
    cap = cv2.VideoCapture(video_path)
    camera = Camera.from_json(camera_config, cap)
    scanner = Scanner.from_json(camera, scaner_config)
    scanner.scan()
    depthmap = scanner.depthmap
    pointcloud = scanner.pointcloud
