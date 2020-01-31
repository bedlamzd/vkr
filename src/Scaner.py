import Camera, Laser


class Scaner:
    def __init__(self, camera: Camera, laser: Laser, distance: float):
        self.d = distance
        self.camera = camera
        self.laser = laser


    def find_coords(self, img):
        pass

    def process_img(self, img):
        pass
