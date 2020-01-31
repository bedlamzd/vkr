from math import tan, cos, sin, pi


class Laser:
    def __init__(self, angle=0, height=0):
        self.angle = angle
        self.height = height
        self.dl = self.laser_to_stripe_dist(height)

    def laser_to_stripe_dist(self, height=None):
        height = height or self.height
        return height * tan(self.angle)
