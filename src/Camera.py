import numpy as np
from typing import Optional
import cv2
from cv2 import VideoCapture
from typing import Tuple, List, Sequence


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
    def from_json_file(cls, filepath: str, cap: Optional[VideoCapture] = None) -> 'Camera':
        # TODO: write json parsing
        mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored = (filepath)
        return Camera(mtx, roi, dist_coef, rot_mtx, tvec, ksize, sigma, threshold, colored, cap)

    @classmethod
    def from_config_file(cls, filepath: str, cap: Optional[VideoCapture] = None) -> 'Camera':
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
            # TODO: Чтение из .ini
            import configparser
