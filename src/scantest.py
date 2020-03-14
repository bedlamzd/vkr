import Scaner
import utilities
import numpy as np
import cv2

fx = 580
fy = fx
u0 = 640 // 2 - 1
v0 = 480 // 2 - 1
roll, pitch, yaw = np.radians([0, 30, 0])

mtx = np.array([[fx, 0, u0],
                [0, fy, v0],
                [0, 0, 1]])
rot_mtx = utilities.roteul(pitch, yaw, roll, order='XYZ')
R = np.array([[0, 0, -1],
              [-1, 0, 0],
              [0, 1, 0]])
rot_mtx = R@rot_mtx
tvec = np.array([200, 112, 152.27])

height = tvec[2] / np.cos(pitch)
angle = pitch
velocity = 300/60
img_proc_opts = {}
extraction_opts = {}

cap = cv2.VideoCapture(r"C:\Users\bedla\YandexDisk\MT.lab\МТ.П - Производство\МТ.П.001\Файлы для скана\scanner (2).mp4")
camera = Scaner.Camera(mtx=mtx, rot_mtx=rot_mtx, tvec=tvec, cap=cap)
scaner = Scaner.Scaner(camera=camera, height=height, angle=angle, velocity=velocity, img_proc_opts=img_proc_opts,
                       extraction_opts=extraction_opts)
scaner.scan()
x, y, z = [x.reshape(x.shape[:2]) for x in np.dsplit(scaner.pointcloud,3)]
xn, yn, zn = [utilities.normalize(x) for x in [x,y,z]]
pointcloud = scaner.pointcloud
np.clip(pointcloud[..., 2], 60, None, out=pointcloud[..., 2])
utilities.show_height_map(pointcloud)
# utilities.show_height_map(scaner.pointcloud)
