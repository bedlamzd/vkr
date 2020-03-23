import numpy as np
from typing import Optional
import cv2
from cv2 import VideoCapture
from typing import Tuple, List, Sequence
import time


# TODO: logging
class Camera:
    """
    A class representing a camera

    :ivar mtx: camera intrinsic matrix [[fx,  0, u0],
                                        [ 0, fy, v0],
                                        [ 0,  0,  1]]
    :type mtx: np.ndarray
    :ivar roi: region of interest in camera view (x, y, w, h)
    :type roi: Sequence
    :ivar dist_coef: distortion coefficients of camera
    :type dist_coef: np.ndarray
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

    def __init__(self,
                 mtx: Optional[np.ndarray] = None,
                 roi: Optional[Sequence] = None,
                 dist_coef: Optional[np.ndarray] = None,
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
        self.dist_coef = dist_coef
        self.rot_mtx = rot_mtx
        self.tvec = tvec
        self.ksize = ksize
        self.sigma = sigma
        self.threshold = threshold
        self.colored = colored
        self._cap = cap  # type: Optional[VideoCapture]
        self._frame_size = 0
        self._frame_width = 0
        self._frame_height = 0

    @classmethod
    def from_json(cls, filepath: str, cap: Optional[VideoCapture] = None) -> 'Camera':
        # TODO: write json parsing
        mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored = (filepath)
        return Camera(mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored, cap)

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
    def roi(self) -> Tuple:
        return (0, 0, *self.frame_size) if self._roi is None else self._roi

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
        # TODO: choose between this setter and self.set_frame_idx
        assert isinstance(idx, int), 'index should be int'
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def set_frame_idx(self, idx: int):
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
        return cv2.undistort(img, self.mtx, self.dist_coef, None, self.optimal_mtx)

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


class Calibrator:
    def __init__(self,
                 board_size,
                 board_coordinates=None,
                 win_size=(5, 5),
                 zero_zone=(-1, -1),
                 flags=None,
                 square_size=1,
                 samples=10,
                 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                 ):

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

        self.mtx = None
        self.dist = None
        self.new_mtx = None
        self.roi = None

    def calibrate_video(self, video: cv2.VideoCapture, delay=1, stream=False):
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
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            current_timestamp = time.time() if stream else camera.frame_timing
            if ret and abs(current_timestamp - last_timestamp) >= delay:
                obj_points.append(self.board_coordinates)
                corners = cv2.cornerSubPix(gray, corners, self.win_size, self.zero_zone, self.criteria)
                img_points.append(corners)
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

    def calibrate_images(self, images: List[np.ndarray]):
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
