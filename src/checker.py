from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional
import numpy as np
import json
from ezdxf.math.vector import Vector
from tools.general import pairwise


@dataclass
class Mark:
    """
    dataclass for storing info on a single mark on a table

           |  width |

           /re----fs\     - height
    ____rs/__________\fe______

          |          |
          |  length  |

    idc: indices corresponding to raise start, raise end, fall start, fall end of a mark in a laser
    coords: coordinates corresponding to raise start, raise end, fall start, fall end of a mark
    height: average of raise end and fall start heights
    length: plane length from raise start to fall end
    width: plane length from raise end to fall start
    rais: plane length of raise section
    fall: plane length of fall section
    """
    idc: Tuple[int]
    coords: Tuple

    def __post_init__(self):
        self.coords = tuple(Vector(coord) for coord in self.coords)  # type: Tuple[Vector]

    @property
    def raise_start(self) -> Vector:
        return self.coords[0]

    @property
    def raise_end(self) -> Vector:
        return self.coords[1]

    @property
    def fall_start(self) -> Vector:
        return self.coords[2]

    @property
    def fall_end(self) -> Vector:
        return self.coords[3]

    @property
    def height(self):
        return (self.raise_end.z + self.fall_start.z) / 2

    @property
    def width(self):
        return self.raise_end.xy.distance(self.fall_start.xy)

    @property
    def length(self):
        return self.raise_start.xy.distance(self.fall_end.xy)

    @property
    def raise_length(self):
        return self.raise_start.xy.distance(self.raise_end.xy)

    @property
    def fall_length(self):
        return self.fall_start.xy.distance(self.fall_end.xy)

    @property
    def center(self):
        return self.raise_end.xy + self.width / 2


@dataclass
class Checkpoint:
    """
    dataclass storing info of a sequence of marks

    marks: all marks in checkpoint in left to right order
    n: number of marks in checkpoint
    gaps: length of gaps between neighbouring checkpoints
    """
    marks: Sequence[Mark]
    n: int = field(init=False)
    gaps: Tuple = field(init=False)

    def __post_init__(self):
        self.n = len(self.marks)
        self.gaps = tuple(mark1.fall_end.xy.distance(mark2.raise_start.xy) for mark1, mark2 in pairwise(self.marks))

    @property
    def ref_coord(self):
        return self.marks[0].center.xy

    def __iter__(self):
        return iter(self.marks)


class Checker:
    # mapping constants which represent impulse values
    _a = 1
    _b = 3
    # mapping of possible sequence impulses to be marks
    _mapping = {_a: ['end'],
                _b: [-_b, -(_a + _b)],
               _a + _b: [-_b, -(_a + _b)],
                -_a: [_a + _b],
                -_b: ['end'],
                -(_a + _b): [_a]}

    _config_attr = [
        'height',
        'widths',
        'gaps',
        'n',
        'tol',
        'gap_tol',
        'height_tol',
        'width_tol'
    ]

    def __init__(self, height, widths, gaps, n, tol=.5,
                 ref_coord=(0, 0, 0), *,
                 gap_tol=None,
                 height_tol=None,
                 width_tol=None):
        self.expected_height = height
        self.expected_widths = widths
        self.expected_gaps = gaps
        self.expected_n = n
        self.gap_tol = gap_tol or 2 * tol
        self.height_tol = height_tol or tol
        self.width_tol = width_tol or 2 * tol
        self.ref_coord = Vector(ref_coord)

    @classmethod
    def load_json(cls, filepath) -> 'Checker':
        data = json.load(open(filepath))
        data = {attr: value for attr, value in data.items() if attr in cls._config_attr}
        return cls(**data)


    def make_sequence(self, coords):
        sequence = np.copy(coords[..., 2])
        sequence[np.abs(sequence) < self.height_tol] = 0
        sequence[(sequence != 0) & (np.abs(sequence - self.expected_height) > self.height_tol)] = -self._a
        sequence[(sequence != -self._a) & (sequence != 0) & (~np.isnan(sequence))] = self._b
        sequence = np.diff(sequence, prepend=0, append=0)[1:-1]
        return sequence

    def make_marks(self, coords):
        sequence = self.make_sequence(coords)
        stack = []
        idc = []
        marks = []
        for idx, item in enumerate(sequence):
            try:
                if not self._mapping.get(item):
                    continue
                elif 'end' in self._mapping.get(item) and stack and item in self._mapping.get(stack[-1]):
                    stack.append(item)
                    idc = idc + [idx, idx] if item == -self._b else idc + [idx]
                    if len(idc) == 2:
                        print()
                    marks.append(Mark(idc, coords[idc]))
                    stack = []
                    idc = []
                    continue
            except IndexError:
                print()
            if (not stack and item in (-self._a, self._b)) or (stack and item in self._mapping.get(stack[-1], [])):
                stack.append(item)
                idc = idc + [idx, idx] if item == self._b else idc + [idx]
            else:
                stack = []
                idc = []
        return marks

    def make_checkpoint(self, coords):
        marks = self.make_marks(coords)
        checkpoint = Checkpoint(marks) if marks else False
        return checkpoint

    def check_gaps(self, checkpoint: Checkpoint):
        if not self.expected_gaps:
            return True
        else:
            actual_gaps = np.array(checkpoint.gaps)
            try:
                gaps_errors = np.abs(actual_gaps - self.expected_gaps)
            except ValueError:  # happens when there is size mismatch
                return False
        return np.all(gaps_errors < self.gap_tol)

    def check_widths(self, checkpoint: Checkpoint):
        if not self.expected_widths:
            return True
        else:
            actual_widths = np.array([mark.width for mark in checkpoint.marks])
            try:
                width_errors = np.abs(actual_widths - self.expected_widths)
            except ValueError:  # happens when there is size mismatch
                return False
        return np.all(width_errors < self.width_tol)

    def check_n(self, checkpoint: Checkpoint):
        """
        check if number of marks match expected quantity if expected at all

        :param checkpoint:
        :return:
        """
        return not self.expected_n or checkpoint.n == self.expected_n

    def check_all(self, checkpoint: Checkpoint):
        return self.check_n(checkpoint) and self.check_widths(checkpoint) and self.check_gaps(checkpoint)

    def __call__(self, coords) -> (bool, Optional[Checkpoint]):
        checkpoint = self.make_checkpoint(coords)
        if checkpoint and self.check_all(checkpoint):
            return True, checkpoint
        else:
            return False, None
