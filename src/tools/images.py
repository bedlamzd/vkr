from functools import wraps

import cv2
import numpy as np
import imutils

from typing import Union, List

from src.tools.general import nothing


def generate_chessboard(square_size=30, grid=(8, 8)):
    chessboard = np.full(np.multiply(square_size, grid), 255, dtype=np.uint8)
    for m in range(grid[0]):
        row_start = square_size * m
        row_stop = square_size * (m + 1)
        for n in range(grid[1]):
            col_start = square_size * (2 * n + (m % 2))
            col_stop = col_start + square_size
            chessboard[row_start:row_stop, col_start:col_stop] = 0
    return chessboard


def show_img(img, winname='image', exit_key=27):
    cv2.namedWindow(winname)  # создать окно
    cv2.imshow(winname, img)  # показать в окне картинку
    # пока окно открыто и кнопка выхода не нажата ждать
    while cv2.getWindowProperty(winname, 0) >= 0 and cv2.waitKey(50) != exit_key:
        pass
    cv2.destroyAllWindows()  # завершить процессы


def find_contours(img: Union[np.ndarray, str]) -> (List, np.ndarray):
    """
    Находит контуры в изображении

    :param img: картинка либо путь до картинки
    :return: возвращает контуры из изображения и изображение с отмеченными на нём контурами
    """
    # проверка параметр строка или нет
    original = None
    gray = None
    if isinstance(img, str):
        original = cv2.imread(img)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            original = img.copy()
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            original = cv2.merge((img.copy(), img.copy(), img.copy()))
            gray = img.copy()
    else:
        raise TypeError(f'передан {type(img)}, ожидалось str или numpy.ndarray')

    # избавление от минимальных шумов с помощью гауссова фильтра и отсу трешхолда
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, gausThresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # нахождение замкнутых объектов на картинке с помощью морфологических алгоритмов
    kernel = np.ones((5, 5), np.uint8)
    # gausThresh = cv2.morphologyEx(gausThresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opening = cv2.morphologyEx(gausThresh, cv2.MORPH_OPEN, kernel, iterations=10)
    # найти однозначный задний фон
    sureBg = cv2.dilate(opening, kernel, iterations=3)
    distTrans = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    # однозначно объекты
    ret, sureFg = cv2.threshold(distTrans, 0.1 * distTrans.max(), 255, 0)
    sureFg = np.uint8(sureFg)
    # область в которой находятся контура
    unknown = cv2.subtract(sureBg, sureFg)
    # назначение маркеров
    ret, markers = cv2.connectedComponents(sureFg)
    # отмечаем всё так, чтобы у заднего фона было точно 1
    markers += 1
    # помечаем граничную область нулём
    markers[unknown == 255] = 0
    markers = cv2.watershed(original, markers)
    # выделяем контуры на изображении
    original[markers == -1] = [0, 0, 255]
    # вырезаем ненужный контур всей картинки
    contours = []
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker] = 255
        tmp = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        tmp = imutils.grab_contours(tmp)
        contour = sorted(tmp, key=cv2.contourArea, reverse=True)[0]
        contours.append(contour)
    blankSpace = np.full(gray.shape, 255, dtype='uint8')
    blankSpace[markers <= 1] = 0
    # применяем на изначальную картинку маску с задним фоном
    result = cv2.bitwise_and(original, original, mask=blankSpace)
    for contour in contours:
        cv2.drawContours(result, [contour], -1, np.random.randint(0, 255, 3).tolist(), 1)
        # show_img(result)
    return contours, result


def find_center_and_rotation(contour, rotation=True):
    """
    Найти центр и поворот контура

    :param contour: контур для расчётв
    :param rotation: находить ли его поворот
    :return:
        центр контура (row, col)
        центр контура (row, col) и его поворот
    """
    moments = cv2.moments(contour)
    center_x = moments['m10'] / moments['m00']  # row
    center_y = moments['m01'] / moments['m00']  # column
    center = (center_x, center_y)
    if rotation:
        a = moments['m20'] / moments['m00'] - center_x ** 2
        b = 2 * (moments['m11'] / moments['m00'] - center_x * center_y)
        c = moments['m02'] / moments['m00'] - center_y ** 2
        theta = 1 / 2 * np.arctan(b / (a - c)) + (a < c) * np.pi / 2
        return center, theta
    else:
        return center


def select_hsv_values(video):
    """
    функция помощник для подбора hsv значений для фильтра
    :param video: видеопоток либо с камеры либо из файла
    :return:
    """
    params = {'h1': 0, 'h2': 255, 's1': 0, 's2': 255, 'v1': 0, 'v2': 255}
    setwin = 'hsv_set'
    reswin = 'result'
    cv2.namedWindow(setwin, cv2.WINDOW_NORMAL)
    cv2.namedWindow(reswin, cv2.WINDOW_NORMAL)
    for key in params:
        cv2.createTrackbar(key, setwin, params[key], 255, nothing)
    cv2.createTrackbar('mask', setwin, 0, 1, nothing)

    # noinspection PyShadowingNames
    def get_params(win='hsv_set'):
        for key in params:
            params[key] = int(cv2.getTrackbarPos(key, win))
        m = int(cv2.getTrackbarPos('mask', setwin))
        hsv_lower = tuple(params[k] for k in ['h1', 's1', 'v1'])
        hsv_upper = tuple(params[k] for k in ['h2', 's2', 'v2'])
        return hsv_lower, hsv_upper, m

    cap = cv2.VideoCapture(video)
    lowerb, upperb = (0, 0, 0), (255, 255, 255)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lowerb, upperb, m = get_params(setwin)
            hsv = cv2.inRange(hsv, lowerb, upperb, None)
            result = cv2.bitwise_and(frame, frame, mask=hsv) if m == 1 else hsv
            cv2.imshow(reswin, result)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ch = cv2.waitKey(15)
        if ch == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return lowerb, upperb


def decor_stream2img(img_func):
    """
    Декоратор позволяющий использовать функции для кадров с видео

    :param img_func: функция работающая только с кадрами
    :return: генератор
    """
    import cv2
    # TODO: consider to delete or replace with custom iterator object for cap (which is actually Camera class)
    #       maybe extend functionality for specific use
    @wraps(img_func)
    def wrapper(video, loops=False, *args, **kwargs):
        """
        Принимает на вход видео к которому покадрово нужно применить функцию

        :param video: видео для обработки
        :param loops: зациклить видео или нет (если video это поток с камеры то False в любом случае)
        :param args: параметры для функции
        :param kwargs: доп параметры и именные аргументы для функции
        :keyword max_loops: максимальное число циклов по видео. default = 10. None - бесконечно
        :return: поочерёдно результат img_func для каждого кадра
        """
        count_loops = 0
        max_loops = kwargs.pop('max_loops', 10)
        cap = cv2.VideoCapture(video)
        if isinstance(video, int):
            loops = False
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                res = img_func(frame, *args, **kwargs)
                yield res
            elif loops:
                if max_loops is None or count_loops < max_loops:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    count_loops += 1
                else:
                    print('max loops reached')
                    cap.release()
            else:
                print('video ended or crashed')
                cap.release()

    return wrapper


def fast_calibration(device, board_size=(6, 4), manual=True, autofocus=False, *args, **kwargs):
    from Camera import Camera

    cam = Camera(cap=cv2.VideoCapture(device))
    if not autofocus:
        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cv2.namedWindow("Video")
        cv2.createTrackbar("Focus", "Video", 0, 500, lambda v: cam.cap.set(cv2.CAP_PROP_FOCUS, v / 10))
    cam.calibrate_intrinsic(board_size=board_size, manual=manual, *args, **kwargs)
    cam.cap.release()
    print(cam.mtx, cam.dist)
    return cam


def fast_pose(device, board_size=(6, 4), autofocus=False, *args, **kwargs):
    from Camera import Camera, CameraCalibrator

    # TODO: make matrix required or positional
    if 'mtx' not in kwargs:
        mtx = np.array([[580, 0, 319],
                        [0, 580, 239],
                        [0, 0, 1]], dtype=np.float)
        kwargs['mtx'] = mtx
    cam = Camera(cap=cv2.VideoCapture(device))
    calibrator = CameraCalibrator(board_size=board_size, *args, **kwargs)
    if not autofocus:
        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cv2.namedWindow("Video")
        cv2.createTrackbar("Focus", "Video", 0, 500, lambda v: cam.cap.set(cv2.CAP_PROP_FOCUS, v / 10))
    for img in cam:
        if cv2.waitKey(15) == 27: break
        ret, rot_mtx, tvec = calibrator.extrinsic_from_image(img)
        if ret:
            print('R = {}\nX = {}\nY = {}\nZ = {}\n'.format(rot_mtx, *(tvec)), end='-' * 10 + '\n')
            print('R = {}\nX = {}\nY = {}\nZ = {}\n'.format(rot_mtx.T, *(rot_mtx.T @ tvec)), end="#" * 10 + '\n')
        cv2.imshow("Video", img)
    cam.cap.release()
    cv2.destroyAllWindows()


def take_shot(device, dir='.', name_template='shot%d'):
    from Camera import Camera

    i = 0
    cam = Camera(cap=cv2.VideoCapture(device))
    for img in cam:
        key = cv2.waitKey(15)
        cv2.imshow("Video", img)
        if key == 27:
            break
        elif key == 13:
            name = f'{dir}/{name_template % i}.png'
            while not cv2.imwrite(name, img): pass
            i += 1
            print(f'Shot taken. {name}')
    cv2.destroyAllWindows()
    cam.cap.release()


def draw_text(img, text: str, org, fontFace, fontScale, color, thickness, spacing=2, **kwargs):
    lines = text.splitlines()
    cv2.putText(img, lines[0], org, fontFace, fontScale, color, thickness, **kwargs)
    text_height = 0
    for line in lines[1:]:
        text_height += spacing + cv2.getTextSize(line, fontFace, fontScale, thickness)[0][1]
        cv2.putText(img, line, (org[0], org[1] + text_height), fontFace, fontScale, color, thickness, **kwargs)
