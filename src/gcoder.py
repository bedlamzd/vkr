import json
from typing import List, Optional

import numpy as np

from cookie import Cookie
from elements import Drawing


class Gcode(list):
    def __init__(self, instructions: Optional[List[str]] = None):
        super(Gcode, self).__init__()
        if instructions is not None: self.extend(instructions)

    def save(self, path: str):
        with open(path, 'w+') as file:
            for line in self:
                file.write(f'{line}\n')
        # TODO: logging instead of print
        print(f'Gcode saved to {path}')

    def comment(self, text):
        """

        :param text: text of a comment
        :return:
        """
        self.append(f'; {text}')

    def _abstract_command(self, code, **kwargs):
        command = f'{code} ' + ' '.join(f'{key}{value:.3f}' for key, value in kwargs.items() if value is not None)
        self.append(command)

    def home(self, **kwargs):
        """

        :param kwargs:
            :keyword O:
            :keyword R:
            :keyword X:
            :keyword Y:
            :keyword Z:
        :return:
        """
        params = ('O', 'R', 'X', 'Y', 'Z')
        self._abstract_command(code='G28', **{key: value for key, value in kwargs.items() if key in params})

    def slow_move(self, **kwargs):
        """

        :param kwargs:
            :keyword X:
            :keyword Y:
            :keyword Z:
            :keyword E:
            :keyword F:
        :return:
        """
        params = ('X', 'Y', 'Z', 'E', 'F')
        self._abstract_command(code='G1', **{key: value for key, value in kwargs.items() if key in params})

    def rapid_move(self, **kwargs):
        """

        :param kwargs:
            :keyword X:
            :keyword Y:
            :keyword Z:
            :keyword E:
            :keyword F:
        :return:
        """
        params = ('X', 'Y', 'Z', 'E', 'F')
        self._abstract_command(code='G0', **{key: value for key, value in kwargs.items() if key in params})

    def set_position(self, **kwargs):
        """

        :param kwargs:
            :keyword X:
            :keyword Y:
            :keyword Z:
            :keyword E:
        :return:
        """
        params = ('X', 'Y', 'Z', 'E',)
        self._abstract_command(code='G92', **{key: value for key, value in kwargs.items() if key in params})


class Gcoder:
    _config_attr = [
        'tolerance',
        'z_offset',
        'extr_coef',
        'extr_mult',
        'retraction',
        'p0',
        'p1',
        'p2',
        'slice_step',
        'F0',
        'F1',
        'point_apprx'
    ]

    def __init__(self,
                 tolerance: float = 0.5,
                 slice_step: float = 1.0,
                 z_offset: float = .0,
                 extr_coef: float = 0.041,
                 extr_mult: float = 1,
                 retraction: float = 2.05,
                 p0: float = 0.05,
                 p1: float = 0.15,
                 p2: float = 0.9,
                 z_max: float = 200.0,
                 point_apprx: str = 'mls',
                 F0: float = 2500,
                 F1: float = 1000):
        # self.gcode = Gcode()  # type: Gcode
        self.tolerance = tolerance
        self.slice_step = slice_step
        self.z_offset = z_offset
        self.point_apprx = point_apprx
        self.apprx_param = dict()

        self.F0 = F0
        self.F1 = F1
        self.extr_coef = extr_coef
        self.extr_mult = extr_mult
        self.retraction = retraction
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.z_max = z_max

    @property
    def config_data(self):
        return {attr: getattr(self, attr) for attr in self._config_attr}

    @classmethod
    def load_json(cls, json_path):
        data = json.load(open(json_path))
        data = {attr: value for attr, value in data.items() if attr in cls._config_attr}
        return cls(**data)

    def dump_json(self, filepath, **kwargs):
        json.dump(self.config_data, open(filepath, 'w'), **kwargs)
        # TODO: logging instead of print
        print(f'Gcoder data saved to {filepath}')

    def generate_gcode(self, drawing: Drawing, cookies: List[Cookie], pointcloud: Optional[np.ndarray] = None, pre_gcode=None):
        drawing.slice(self.slice_step)  # prepare drawing
        E = 0
        if pre_gcode is not None:
            gcode = Gcode(pre_gcode)
        else:
            gcode = Gcode()
            gcode.home()
        gcode.set_position(E=E)
        gcode.rapid_move(Z=self.z_max, F=self.F0)
        for count, cookie in enumerate(cookies, 1):
            # TODO: logging instead of print
            print(f'Processing cookie #{count}...')
            z_up = cookie.max_height + 5
            if z_up > self.z_max: z_up = self.z_max
            gcode.comment(f'{count:3d} cookie')
            drawing.center = cookie.center[:2]
            drawing.rotation = cookie.rotation
            drawing.add_z(cookie.pointcloud if pointcloud is None else pointcloud, point_apprx=self.point_apprx, **self.apprx_param)
            for layer_index, layer in enumerate(sorted(drawing.layers.values(), key=lambda x: x.priority)):
                # TODO: logging instead of print
                print(f'    Processing layer #{layer_index}...')
                gcode.comment(f'{layer_index:3d} layer: {layer.name} in drawing')
                if layer.name == 'Contour':
                    gcode.comment(f'    Contour layer. skipped')
                    # TODO: logging instead of print
                    print(f'Layer skipped. Name: {layer.name}; Priority: {layer.priority}')
                    continue
                for contour_index, contour in enumerate(layer.contours):
                    # TODO: logging instead of print
                    print(f'        Processing contour #{contour_index}...')
                    printed_length = 0
                    delta_e = self.extr_coef * self.extr_mult
                    gcode.comment(f'    {contour_index:3d} contour in layer')
                    gcode.rapid_move(X=contour.first_point.x, Y=contour.first_point.y, Z=z_up)
                    gcode.rapid_move(Z=contour.first_point.z + self.z_offset)
                    gcode.slow_move(F=self.F1)
                    last_point = contour.first_point
                    for element_index, element in enumerate(contour.elements, 1):
                        gcode.comment(f'        {element_index:3d} element in contour')
                        for point in element.get_points()[1:]:
                            dL = point.distance(last_point)
                            E += round(delta_e * dL, 3)
                            gcode.slow_move(X=point.x, Y=point.y, Z=point.z + self.z_offset, E=E)
                            printed_length += dL
                            last_point = point
                            printed_percent = printed_length / contour.length
                            if printed_percent < self.p0:
                                delta_e = self.extr_coef * self.extr_mult
                            elif printed_percent < self.p1:
                                delta_e = 0
                            elif printed_percent < self.p2:
                                delta_e = self.extr_coef
                            else:
                                delta_e = 0
                    gcode.rapid_move(F=self.F0)
                    gcode.rapid_move(Z=z_up)
            # TODO: logging instead of print
            print(f'Cookie #{count} done.\n')
        gcode.rapid_move(Z=self.z_max)
        gcode.home()
        # TODO: logging instead of print
        print('Gcode ready')
        return gcode

