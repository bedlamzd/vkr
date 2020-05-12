import Scanner
import numpy as np
import cv2
import time
import random
import imutils

from tools.images import show_img
from tools.general import normalize, avg
from tools.pointcloud import show_height_map
from tools.math import roteul, rotx, roty, rotz

## подготовка данных
# матрица внутренних параметров камеры
fx = 580
fy = fx
u0 = 640 // 2 - 1
v0 = 480 // 2 - 1
mtx = np.array([[fx, 0, u0],
                [0, fy, v0],
                [0, 0, 1]])

# область интереса в кадре
roi = [30, 100, 600, 300]

# матрица поворота и вектор перемещения камеры. если матрица относительно мира, то необходимо найти обратное преобразование
rot_mtx = rotx(-np.pi / 6)
# rot_mtx = np.array([[0.99859376, 0.03779394, 0.03717441],
#                     [-0.01415901, 0.86591157, -0.49999658],
#                     [-0.05108658, 0.49876718, 0.86522901]]).T
# rot_mtx = np.eye(3)
tvec = np.array([250, 112, 168.07156932])
# T = np.linalg.inv(np.row_stack([np.column_stack([rot_mtx, tvec]), [0, 0, 0, 1]]))  # обратное преобразование
# rot_mtx, tvec = T[:3, :3], T[:3, 3]
R = np.array([[0, -1, 0],  # если координатные оси не совпдают с выбранными, привести к реальным
              [-1, 0, 0],
              [0, 0, -1]])
rot_mtx = R @ rot_mtx
rot_mtx = rot_mtx

# параметры сканера
height = 171.45
angle = np.radians(-39)
velocity = np.array([-300 / 60, 0, 0])  # вектор скорости перемещения камеры
img_proc_opts = {}
extraction_mode = 'ggm'
extraction_opts = {'ksize': 29, 'sigma': 4.45}

# Формирование итоговых объектов из данных
cap = cv2.VideoCapture(r"C:\Users\bedla\YandexDisk\Диплом Борисов\Иллюстрации\Видео с камеры.mp4")
camera = Scanner.Camera(mtx=mtx, rot_mtx=rot_mtx, tvec=tvec, cap=cap, roi=roi)
scaner = Scanner.Scanner(camera=camera, height=height, angle=angle, velocity=velocity, img_proc_opts=img_proc_opts,
                         extraction_opts=extraction_opts, extraction_mode=extraction_mode)
scalibrator = scanner.ScannerCalibrator(scaner)

## уравнение координат
# расчёт в СК камеры
z_k = lambda y: height * np.tan(angle) / (np.tan(angle) + (y - v0) / fx)
y_k = lambda y: z_k(y) * (y - v0) / fy
x_k = lambda x: z_k(x) * (x - u0) / fx

# расчёт в СК мира
x_w = lambda x, y: x_k(x) * (rot_mtx[0, 0] * (x - u0) + rot_mtx[0, 1] * (y - v0) / fy + rot_mtx[0, 2]) + tvec[0]
y_w = lambda x, y: y_k(y) * (rot_mtx[1, 0] * (x - u0) + rot_mtx[1, 1] * (y - v0) / fy + rot_mtx[1, 2]) + tvec[1]
z_w = lambda x, y: z_k(y) * (rot_mtx[2, 0] * (x - u0) + rot_mtx[2, 1] * (y - v0) / fy + rot_mtx[2, 2]) + tvec[2]


def test_scan():
    scaner.scan()
    pointcloud = scaner.pointcloud
    # pointcloud[..., 2] -= 74
    x, y, z = [x.reshape(x.shape[:2]) for x in np.dsplit(pointcloud, 3)]
    xn, yn, zn = [normalize(x) for x in [x, y, z]]
    # np.clip(pointcloud[..., 0], -250, 200, out=pointcloud[..., 0])
    # np.clip(pointcloud[..., 1], -250, 200, out=pointcloud[..., 1])
    # np.clip(pointcloud[..., 2], 0, 200, out=pointcloud[..., 2])
    show_height_map(pointcloud)


def test_scan_artificial():
    rows, cols = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    out = cv2.VideoWriter('test_scan.mp4', fourcc, 10, (cols, rows))
    for i in range(479, 238, -1):
        img = np.zeros((rows, cols), dtype=np.uint8)
        img[i] = 255
        out.write(img)
    out.release()
    camera.cap = cv2.VideoCapture('test_scan.mp4')
    camera._roi = None
    scaner._cloud = np.zeros((camera.frame_count, camera.frame_width, 3))
    scaner.scan()
    show_height_map(scaner.pointcloud)


def test_scan_calibration_artificial(randomize=False, ampl=1, shift=-0.5):
    images = []
    heights = []
    rows = (v0, 230, 220, 100)
    real_data = []
    for i in rows:
        if randomize:
            y = i + ampl * random.random() + shift
            pxl = i
        else:
            y = pxl = i
        heights.append(z_w(u0, y) - z_w(u0, rows[0]))
        real_data.append((pxl, y, heights[-1]))
        img = np.zeros((480, 640), dtype=np.uint8)
        img[pxl] = 255
        images.append(img)
    for pixel, y, height in real_data: print(f'pixel = {pixel:6.2f}, y = {y:6.2f}, height = {height:6.2f}')

    scalibrator = scanner.ScannerCalibrator(scaner)
    scalibrator.calibrate_from_images(images, heights)


def test_scan_calibration_artificial2(randomize=False, ampl=1, shift=-0.5):
    images = []
    z_coordinates = []
    rows = (v0, 230, 220, 100)
    real_data = []
    for i in rows:
        if randomize:
            y = i + ampl * random.random() + shift
            pxl = i
        else:
            y = pxl = i
        z_coordinates.append(z_w(u0, y))
        real_data.append((pxl, y, z_coordinates[-1]))
        img = np.zeros((480, 640), dtype=np.uint8)
        img[pxl] = 255
        images.append(img)
    for pixel, y, height in real_data: print(f'pixel = {pixel:6.2f}, y = {y:6.2f}, height = {height:6.2f}')

    scalibrator.calibrate_from_images2(images, z_coordinates)


def test_scan_calibration_real():
    images = [cv2.imread(f'C:/Users/bedla/Repositories/vkr/shot{i}.png', 0) for i in range(4)]

    scalibrator.calibrate_from_images2(images, [0, 9, 28, 42])


def test_camera_calibration(device, board_size=(6, 4), manual=True, autofocus=False, *args, **kwargs):
    from camera import Camera

    cam = Camera(cap=cv2.VideoCapture(device))
    if not autofocus:
        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cv2.namedWindow("Video")
        cv2.createTrackbar("Focus", "Video", 0, 500, lambda v: cam.cap.set(cv2.CAP_PROP_FOCUS, v / 10))
    cam.calibrate_intrinsic(board_size=board_size, manual=manual, *args, **kwargs)
    cam.cap.release()
