import cv2

import globalValues
from Camera import Camera
from Scanner import Scanner
from StartDetector import Checker
from cookie import find_cookies, process_cookies
from elements import Drawing
from gcoder import Gcoder


def main(video_file: str,
         path_to_dxf: str,
         camera_json: str,
         checker_json: str,
         scanner_json: str,
         gcoder_json: str,
         gcode_path: str):

    # Initiate objects
    cap = cv2.VideoCapture(video_file)
    dwg = Drawing.from_file(dxf_path=path_to_dxf)
    camera = Camera.load_json(filepath=camera_json, cap=cap)
    checker = Checker.load_json(filepath=checker_json)
    scanner = Scanner.load_json(filepath=scanner_json, camera=camera, checker=checker)
    gcoder = Gcoder.load_json(gcoder_json)

    # Acquire depth map and pointcloud
    scanner.scan()
    scanner.bind_coordinates()
    depthmap, pointcloud = scanner.depthmap, scanner.pointcloud

    # Acquire cookies position
    cookies, detected_contours = find_cookies(depthmap, pointcloud)
    cookies, detected_contours = process_cookies(cookies, pointcloud, detected_contours)

    # Generate and save Gcode
    gcode = gcoder.generate_gcode(drawing=dwg, cookies=cookies)
    gcode.save(gcode_path)
