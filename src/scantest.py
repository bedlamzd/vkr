import Scanner
import numpy as np
import cv2
import time

fx = 580
fy = fx
u0 = 640 // 2 - 1
v0 = 480 // 2 - 1
roll, pitch, yaw = np.radians([0, 30, 0])
roi = [30, 100,600, 300]

mtx = np.array([[fx, 0, u0],
                [0, fy, v0],
                [0, 0, 1]])
rot_mtx = utilities.roteul(pitch, yaw, roll, order='XYZ')
R = np.array([[0, 0, -1],
              [-1, 0, 0],
              [0, 1, 0]])
rot_mtx = R @ rot_mtx
tvec = np.array([200, 112, 152.27])

height = tvec[2] / np.cos(pitch)
angle = pitch
velocity = 300 / 60
img_proc_opts = {}
extraction_mode = 'ggm'
extraction_opts = {'ksize': 29, 'sigma': 4.45}

cap = cv2.VideoCapture(r"C:\Users\bedla\YandexDisk\Диплом Борисов\Иллюстрации\Видео с камеры.mp4")
camera = Scanner.Camera(mtx=mtx, rot_mtx=rot_mtx, tvec=tvec, cap=cap, roi=roi)
scaner = Scanner.Scanner(camera=camera, height=height, angle=angle, velocity=velocity, img_proc_opts=img_proc_opts,
                         extraction_opts=extraction_opts, extraction_mode=extraction_mode)

if __name__ == '__main__':
    start = time.time()
    scaner.scan()
    print(time.time()-start)
    pointcloud = scaner.pointcloud
    # pointcloud[..., 2] = pointcloud[..., 2] - np.abs(pointcloud[:,0,2].reshape(pointcloud.shape[0], 1))
    x, y, z = [x.reshape(x.shape[:2]) for x in np.dsplit(pointcloud,3)]
    xn, yn, zn = [utilities.normalize(x) for x in [x,y,z]]
    np.clip(pointcloud[..., 2], 0, None, out=pointcloud[..., 2])
    # utilities.show_height_map(pointcloud)
