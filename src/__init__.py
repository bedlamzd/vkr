import cv2

import globalValues
from Camera import Camera
from Scanner import Scanner
from StartDetector import Checker
from cookie import find_cookies, process_cookies
from dxf2gcode import dxf2gcode


def main(video_file: str,
         camera_json: str,
         checker_json: str,
         scanner_json: str,
         path_to_dxf: str):
    # Initiate objects
    cap = cv2.VideoCapture(video_file)
    camera = Camera.load_json(filepath=camera_json, cap=cap)
    checker = Checker.load_json(filepath=checker_json)
    scanner = Scanner.load_json(filepath=scanner_json, camera=camera, checker=checker)

    # Acquire depth map and pointcloud
    scanner.scan()
    scanner.bind_coordinates()
    depthmap, pointcloud = scanner.depthmap, scanner.pointcloud

    # Acquire cookies position
    cookies, detected_contours = find_cookies(depthmap, pointcloud)
    cookies, detected_contours = process_cookies(cookies, pointcloud, detected_contours)

    # Save data to external file TODO: SHOULD BE REWRITTEN
    globalValues.cookies = cookies if cookies else None
    globalValues.height_map = pointcloud

    # Generate and save Gcode TODO: ALSO SHOULD BE REWRITTEN
    dxf2gcode(path_to_dxf=path_to_dxf)
