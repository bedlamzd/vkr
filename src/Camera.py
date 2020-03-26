import numpy as np
from typing import Optional
import cv2
from cv2 import VideoCapture
from typing import Tuple, List, Sequence
import time
import json


def find_angle_of_view(view_range: int,
                       focal: float, *,
                       pxl_size: float = 1) -> float:
    """
    расчитывает угол обзора камеры

    :param view_range: длинна обзора в пикселях
    :param focal: фокусное расстояние
    :param pxl_size: размер пикселя на матрице
    :return: угол в радианах
    """
    # TODO: make it static method
    return 2 * np.arctan(view_range * pxl_size / 2 / focal)


def find_camera_angle(view_width: float,  # mm
                      frame_width: int,  # pxl
                      camera_height: float,  # mm
                      focal: float, *,  # pxl|mm
                      pxl_size: float = 1  # 1 if focal in pxl else proper coefficient
                      ) -> float:
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
    # TODO: make it static method
    view_angle = find_angle_of_view(frame_width, focal, pxl_size=pxl_size)
    cos_camera_angle = 2 * camera_height / view_width * np.tan(view_angle / 2)
    camera_angle = np.arccos(cos_camera_angle)
    return camera_angle


def xyz2uv(xyz, mtx, rot_mtx=None, t_vec=None, *, T=None):
    """
    Given camera intrinsic matrix calculate image plane coordinates of a real 3d point
    :param xyz: coordinates of a point in WSC
    :param mtx: camera intrinsic matrix
    :param rot_mtx: rotation matrix of a camera relative to WCS
    :param t_vec: translation vector of a camera relative to WCS
    :param T: full transformation from WCS to camera coordinate space (extrinsic matrix)
    :return: uv coordinates in image plane
    """
    # TODO: make it static method
    xyz = np.r_[xyz[:3], 1]
    if T is None:
        if rot_mtx is None:
            rot_mtx = np.eye(3)
        if t_vec is None:
            t_vec = np.zeros(3)
        T = np.column_stack([rot_mtx, t_vec])
    uv = mtx @ T @ xyz
    uv = uv / uv[2]
    return uv[:2]


# TODO: logging
# TODO: iterator for camera by frames
# TODO: dump to json
class Camera:
    """
    A class representing a camera

    :ivar mtx: camera intrinsic matrix [[fx,  0, u0],
                                        [ 0, fy, v0],
                                        [ 0,  0,  1]]
    :type mtx: np.ndarray
    :ivar roi: region of interest in camera view (x, y, w, h)
    :type roi: Sequence
    :ivar dist: distortion coefficients of camera
    :type dist: np.ndarray
    :ivar rot_mtx: rotation matrix from camera coordinate system (CCS) to global
    :type rot_mtx: np.ndarray
    :ivar tvec: translation vector, camera coordinates in GCS
    :type tvec: np.ndarray
    :ivar ksize: kernel size for gaussian blur
    :type ksize: int
    :ivar sigma: sigma parameter for gaussian blur
    :type sigma: float
    :ivar threshold: threshold value
    :type threshold: float
    """

    _config_attr = [
        'mtx',
        'dist',
        'roi',
        'rot_mtx',
        'tvec',
        'ksize',
        'sigma',
        'threshold',
        'colored'
    ]

    def __init__(self,
                 mtx: Optional[np.ndarray] = None,
                 roi: Optional[Sequence] = None,
                 dist: Optional[np.ndarray] = None,
                 rot_mtx: Optional[np.ndarray] = None,
                 tvec: Optional[np.ndarray] = None,
                 ksize: int = 3,
                 sigma: float = 0,
                 threshold: float = 0,
                 colored: bool = False,
                 cap: Optional[VideoCapture] = None):
        self.mtx = mtx
        self.optimal_mtx = mtx
        self._roi = roi  # (x, y, w, h)
        self.dist = dist
        self.rot_mtx = rot_mtx
        self.tvec = tvec
        self.ksize = ksize
        self.sigma = sigma
        self.threshold = threshold
        self.colored = colored
        self._cap = cap  # type: Optional[VideoCapture]

    @property
    def config_data(self):
        return {attribute: getattr(self, attribute).tolist() if hasattr(getattr(self, attribute), 'tolist')
        else getattr(self, attribute) for attribute in self._config_attr}

    @classmethod
    def from_json(cls, filepath: str = 'src/camera.json', cap: Optional[VideoCapture] = None) -> 'Camera':
        data = json.load(open(filepath))
        data = {attr: np.array(value) if isinstance(value, Sequence) else value
                for attr, value in data.items() if attr in cls._config_attr}
        return cls(cap=cap, **data)

    def dump_json(self, filepath='src/camera.json', **kwargs):
        json.dump(self.config_data, open(filepath, 'w'), **kwargs)
        print(f'Camera data saved to {filepath}')

    def dumps_json(self, **kwargs):
        return json.dumps(self.config_data, **kwargs)

    @classmethod
    def from_config(cls, filepath: str, cap: Optional[VideoCapture] = None) -> 'Camera':
        # TODO: write json parsing
        mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored = (filepath)
        return Camera(mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored, cap)

    @staticmethod
    def check_parameters(mtx: Optional[np.ndarray] = None,
                         roi: Optional[Sequence] = None,
                         dist_coef: Optional[np.ndarray] = None,
                         rot_mtx: Optional[np.ndarray] = None,
                         tvec: Optional[np.ndarray] = None,
                         ksize: int = 3,
                         sigma: float = 0,
                         threshold: float = 0,
                         colored: bool = False,
                         cap: Optional[VideoCapture] = None):
        # TODO: write parameters validation
        # TODO: Sequence -> numpy.ndarray conversion
        assert mtx is None or (isinstance(mtx, np.ndarray) and mtx.shape == (3, 3))
        assert dist_coef is None or isinstance(dist_coef, np.ndarray)
        assert rot_mtx is None or (isinstance(rot_mtx, np.ndarray) and rot_mtx.shape == (3, 3))
        assert tvec is None or (isinstance(tvec, np.ndarray) and tvec.shape == (3,))
        assert cap is None or isinstance(cap, VideoCapture)

    @property
    def u0(self) -> float:
        """
        Horizontal coordinate of principal point
        """
        assert self.mtx is not None, 'intrinsic matrix is not assigned'
        return self.mtx[0, 2]

    @property
    def v0(self) -> float:
        """
        Vertical coordinate of principal point
        """
        assert self.mtx is not None, 'intrinsic matrix is not assigned'
        return self.mtx[1, 2]

    @property
    def fx(self) -> float:
        """
        Focal length in x-pixels measure
        """
        assert self.mtx is not None, 'intrinsic matrix is not assigned'
        return self.mtx[0, 0]

    @property
    def fy(self) -> float:
        """
        Focal length in y-pixel measure
        """
        assert self.mtx is not None, 'intrinsic matrix is not assigned'
        return self.mtx[1, 1]

    @property
    def focal_length(self) -> Tuple[float, float]:
        """
        Focal length in both pixel measures
        """
        return self.fx, self.fy

    @property
    def roi(self) -> Optional[Tuple]:
        try:
            return self._roi if self._roi else (0, 0, *self.frame_size)
        except AssertionError:
            return None

    @property
    def extrinsic_mtx(self):
        return np.c_[self.rot_mtx, self.tvec]

    @property
    def cap(self) -> VideoCapture:
        """
        Video assosiated with camera
        """
        assert isinstance(self._cap, VideoCapture), 'cap is not assigned'
        return self._cap  # type: VideoCapture

    @cap.setter
    def cap(self, cap: VideoCapture):
        assert isinstance(cap, VideoCapture)
        self._cap = cap

    @property
    def frame_timing(self) -> float:
        """
        Frame timing in seconds
        """
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    @property
    def current_frame_idx(self):
        """
        Index of current frame in video (-1 if not started)
        """
        return self.next_frame_idx - 1

    @property
    def next_frame_idx(self):
        """
        Index of next frame in video
        """
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @next_frame_idx.setter
    def next_frame_idx(self, idx: int):
        """
        Set index of next frame to read
        :param int idx: next frame index
        """
        assert isinstance(idx, int), 'index should be int'
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    @property
    def frame_width(self) -> int:
        """
        Video frame width
        """
        # TODO: записывать эти параметры в _frame_* и возвращать их?
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """
        Video frame height
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_size(self) -> Tuple[int, int]:
        """
        Video resolution
        """
        return self.frame_width, self.frame_height

    @property
    def fps(self) -> float:
        """
        Video framerate
        """
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """
        Total frames in video
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_mask(self, img: np.ndarray) -> np.ndarray:
        if self.threshold == 0:
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold > 0:
            _, mask = cv2.threshold(img, 0, self.threshold, cv2.THRESH_BINARY)
        else:
            mask = np.full_like(img, 255, np.uint8)
        return mask

    def apply_blur(self, img: np.ndarray, *, ksize=None, sigma=None) -> np.ndarray:
        if ksize is None:
            ksize = self.ksize
        if sigma is None:
            sigma = self.sigma
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    @staticmethod
    def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(img, img, mask=mask)

    def apply_roi(self, img: np.ndarray, *, roi=None) -> np.ndarray:
        if roi is None:
            roi = self.roi
        (x, y, w, h) = roi
        return img[y:y + h, x:x + w].copy()

    def apply_color_filt(self, img: np.ndarray, *, color_filt_vals=None) -> np.ndarray:
        if self.colored:
            # TODO: Some RGB/HSV processing based on color_filt_vals
            pass
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.optimal_mtx)

    def process_img(self, img: np.ndarray, *,
                    undistort=False,
                    roi=True,
                    color_filt=True,
                    mask=True) -> np.ndarray:
        new_img = img.copy()
        if undistort:
            new_img = self.undistort(new_img)
        if roi:
            new_img = self.apply_roi(new_img)
        if color_filt:
            new_img = self.apply_color_filt(new_img)
        if mask:
            new_img_blur = self.apply_blur(new_img)
            new_img_mask = self.get_mask(new_img_blur)
            new_img = self.apply_mask(new_img, new_img_mask)
        return new_img

    def read_raw(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()

    def read_processed(self, **kwargs) -> Tuple[bool, np.ndarray]:
        ret, img = self.read_raw()
        if ret:
            img = self.process_img(img, **kwargs)
        return ret, img

    def calibrate_intrinsic(self, delay=1, stream=False, manual=False, **kwargs):
        calibrator = CameraCalibrator(**kwargs)
        self.mtx, self.dist = calibrator.intrinsic_from_video(self.cap, delay, stream, manual)

    def get_new_camera_mtx(self, alpha=1):
        self.optimal_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.frame_size, alpha)
        self.next_frame_idx = 0


class CameraCalibrator:
    # TODO: extrinsic camera calibration
    # TODO: scaner calibration
    def __init__(self,
                 board_size,
                 board_coordinates=None,
                 win_size=(5, 5),
                 zero_zone=(-1, -1),
                 flags=None,
                 square_size=1,
                 samples=10,
                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                 *,
                 mtx=None,
                 dist=None,
                 new_mtx=None,
                 roi=None,
                 rot_mtx=np.eye(3),
                 t_vec=np.zeros(3)):

        self.flags = flags
        self.zero_zone = zero_zone
        self.win_size = win_size
        self.criteria = criteria
        self.samples = samples

        self.board_size = board_size
        if not board_coordinates:
            # if no coordinates provided assume square size as unit length and zero z-coordinate
            board_coordinates = np.zeros((np.prod(board_size), 3), np.float32)
            board_coordinates[:, :2] = np.mgrid[:board_size[0], :board_size[1]].T.reshape(-1, 2) * square_size
        self.board_coordinates = board_coordinates

        self.mtx = mtx
        self.dist = dist
        self.new_mtx = new_mtx
        self.roi = roi
        self.rvec = rot_mtx
        self.tvec = t_vec

    @property
    def extrinsic_matrix(self):
        if self.rvec and self.tvec:
            return np.column_stack([self.rvec, self.tvec])

    def intrinsic_from_video(self, video: cv2.VideoCapture, delay=1, stream=False, manual=False):
        cv2.namedWindow('Calibration')
        camera = Camera(cap=video)
        good_samples = 0
        obj_points = [  # 3d points in real world
            # [first image points], [second image points], ...
            # where points identical to board_coordinates
        ]
        img_points = [  # 2d points in image plane
            # [first image points], [second image points], ...
            # where points are corners coordinates in pixels for each image
        ]
        last_timestamp = -np.inf
        frame_read, frame = camera.read_raw()
        while frame_read and good_samples < self.samples:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None, flags=cv2.CALIB_CB_FAST_CHECK)
            current_timestamp = time.time() if stream else camera.frame_timing
            shot_condition = cv2.waitKey(15) == 13 if manual else abs(current_timestamp - last_timestamp) >= delay
            if ret and shot_condition:
                obj_points.append(self.board_coordinates)
                corners = cv2.cornerSubPix(gray, corners, self.win_size, self.zero_zone, self.criteria)
                img_points.append(corners)
                frame = 255 - frame  # indicate successful shot with negative image
                good_samples += 1
                last_timestamp = current_timestamp
                print(f'Sample taken. Current samples: {good_samples}')
            if cv2.waitKey(15) == 27:
                break
            cv2.drawChessboardCorners(frame, self.board_size, corners, ret)
            cv2.imshow('Calibration', frame)
            frame_read, frame = camera.read_raw()
        else:
            if good_samples == 0:
                print('No good samples. Provide better data.')
                print('Calibration failed.')
            else:
                if good_samples < self.samples: print(
                    'Fewer good samples taken than requested, calibration might be inaccurate.')
                ret, self.mtx, self.dist, *_ = cv2.calibrateCamera(obj_points, img_points, camera.frame_size, None,
                                                                   None)
                print('Calibration done.')
        cv2.destroyAllWindows()
        return self.mtx, self.dist

    def intrinsic_from_images(self, images: List[np.ndarray]):
        obj_points = [  # 3d points in real world
            # [first image points], [second image points], ...
            # where points identical to board_coordinates
        ]
        img_points = [  # 2d points in image plane
            # [first image points], [second image points], ...
            # where points are corners coordinates in pixels for each image
        ]
        good_samples = 0
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None, flags=self.flags)
            if ret:
                obj_points.append(self.board_coordinates)
                corners = cv2.cornerSubPix(gray, corners, self.win_size, self.zero_zone, self.criteria)
                img_points.append(corners)
                good_samples += 1
                print(f'Sample taken. Current samples: {good_samples}')
        if good_samples == 0:
            print('No good samples. Provide better data.')
            print('Calibration failed.')
        else:
            if good_samples < self.samples: print(
                'Fewer good samples taken than requested, calibration might be inaccurate.')
            ret, self.mtx, self.dist, *_ = cv2.calibrateCamera(obj_points, img_points, images[0].size, None, None)
            print('Calibration done.')
        return self.mtx, self.dist

    def get_new_camera_mtx(self, image_size, alpha=1):
        self.new_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, image_size, alpha)
        return self.new_mtx, self.roi

    def extrinsic_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rot_mtx, t_vec = None, None
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, flags=cv2.CALIB_CB_FAST_CHECK)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, self.win_size, self.zero_zone, self.criteria)
            cv2.drawChessboardCorners(image, self.board_size, corners, ret)
            ret, rot_mtx, t_vec = cv2.solvePnP(self.board_coordinates, corners, self.mtx, self.dist)
        return ret, rot_mtx, t_vec

    def calibrate_camera_extrinsic_from_images(self, images):
        rot_mtx = np.zeros((3, 3))
        t_vec = np.zeros(3)
        count = 0
        for image in images:
            ret, R, T = self.extrinsic_from_image(image)
            if ret:
                count += 1
                rot_mtx, t_vec = rot_mtx + R, t_vec + T
        try:
            rot_mtx, t_vec = rot_mtx / count, t_vec / count
        except ZeroDivisionError:
            print('Calibration failed. Zero proper images found.')
        return rot_mtx, t_vec
