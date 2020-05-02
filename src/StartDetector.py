from dataclasses import dataclass, field
from typing import Tuple, Sequence, Optional
import numpy as np
import json
from ezdxf.math.vector import Vector


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
    height: float = field(init=False)
    length: float = field(init=False)
    width: float = field(init=False)
    rais: float = field(init=False)
    fall: float = field(init=False)

    def __post_init__(self):
        self.coords = tuple(Vector(coord) for coord in self.coords)
        self.height = (self.coords[1].z + self.coords[-2].z) / 2
        self.length = self.coords[0].xy.distance(self.coords[-1].xy)
        self.width = self.coords[1].xy.distance(self.coords[-2].xy)
        self.rais = self.coords[0].xy.distance(self.coords[1].xy)
        self.fall = self.coords[-2].xy.distance(self.coords[-1].xy)


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
        self.gaps = tuple(
            self.marks[i - 1].coords[-1].xy.distance(self.marks[i].coords[0].xy) for i in range(1, self.n))


class Checker:
    # TODO: testing
    # mapping constants which represent impulse values
    a = 1
    b = 3
    # mapping of possible sequence impulses to be marks
    mapping = {a: ['end'],
               b: [-b, -(a + b)],
               a + b: [-b, -(a + b)],
               -a: [a + b],
               -b: ['end'],
               -(a + b): [a]}

    _config_attr = [
        'height',
        'width',
        'gaps',
        'n',
        'tol',
        'gap_tol',
        'height_tol',
        'width_tol'
    ]

    def __init__(self, height, widths, gaps, n, tol=.5, *, gap_tol=None, height_tol=None, width_tol=None):
        self.expected_height = height
        self.expected_widths = widths
        self.expected_gaps = gaps
        self.expected_n = n
        self.gap_tol = gap_tol or 2 * tol
        self.height_tol = height_tol or tol
        self.width_tol = width_tol or tol

    @classmethod
    def load_json(cls, filepath) -> 'Checker':
        data = json.load(open(filepath))
        data = {attr: value for attr, value in data.items() if attr in cls._config_attr}
        return cls(**data)


    def make_sequence(self, coords):
        sequence = np.copy(coords[..., 2])
        sequence[np.abs(sequence) < self.height_tol] = 0
        sequence[(sequence != 0) & (np.abs(sequence - self.expected_height) > self.height_tol)] = -self.a
        sequence[sequence > 0] = self.b
        return np.diff(sequence, prepend=0, append=0)[1:-1]

    def make_marks(self, coords):
        sequence = self.make_sequence(coords)
        stack = []
        idc = []
        marks = []
        for idx, item in enumerate(sequence):
            try:
                if not self.mapping.get(item):
                    continue
                elif 'end' in self.mapping.get(item) and stack and item in self.mapping.get(stack[-1]):
                    stack.append(item)
                    idc = idc + [idx, idx] if item == -self.b else idc + [idx]
                    if len(idc) == 2:
                        print()
                    marks.append(Mark(idc, coords[idc]))
                    stack = []
                    idc = []
                    continue
            except IndexError:
                print()
            if (not stack and item in (-self.a, self.b)) or (stack and item in self.mapping.get(stack[-1], [])):
                stack.append(item)
                idc = idc + [idx, idx] if item == self.b else idc + [idx]
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

    def __call__(self, coords):
        checkpoint = self.make_checkpoint(coords)
        if checkpoint and self.check_all(checkpoint):
            return True, checkpoint
        else:
            return False, None


def checker(coords, height, width=None, gaps=None, n: int = None, tol: float = 0.5, *, a: int = 1, b: int = 3,
            **kwargs) -> (bool, Checkpoint):
    gap_tol = kwargs.get('gap_tol', 2 * tol)
    height_tol = kwargs.get('height_tol', tol)
    width_tol = kwargs.get('width_tol', tol)

    m = {a: ['end'], b: [-b, -(a + b)], a + b: [-b, -(a + b)], -a: [a + b], -b: ['end'], -(a + b): [a]}

    def gaps_check(checkpoint: Checkpoint, gaps, gap_tol):
        if isinstance(gaps, (int, float)) or gaps is None:
            n = checkpoint.n
        elif isinstance(gaps, np.ndarray):
            n = gaps.size
        else:
            raise TypeError('width is either a number or 1D ndarray')
        return not gaps or (n == checkpoint.n and np.all(np.abs(np.array(checkpoint.gaps) - gaps) < gap_tol))

    def width_check(checkpoint: Checkpoint, width, width_tol):
        if isinstance(width, (int, float)) or width is None:
            n = checkpoint.n
        elif isinstance(width, np.ndarray):
            n = width.size
        else:
            raise TypeError('width is either a number or 1D ndarray')
        return not width or (n == checkpoint.n and np.all(
            np.abs(np.array([mark.width for mark in checkpoint.marks]) - width) < width_tol))

    def n_check(checkpoint: Checkpoint, n):
        return not n or checkpoint.n == n

    def full_check(checkpoint: Checkpoint, gaps, width, n, gap_tol, width_tol):
        return gaps_check(checkpoint, gaps, gap_tol) and width_check(checkpoint, width, width_tol) and n_check(
            checkpoint, n)

    def make_sequence(coords, height, height_tol):
        sequence = np.copy(coords[..., 2])
        sequence[np.abs(sequence) < height_tol] = 0
        sequence[(sequence != 0) & (np.abs(sequence - height) > height_tol)] = -a
        sequence[sequence > 0] = b
        return np.diff(sequence, prepend=0, append=0)[1:-1]

    def process_sequence(sequence, coords):
        stack = []
        idc = []
        marks = []
        for idx, item in enumerate(sequence):
            try:
                if not m.get(item):
                    continue
                elif 'end' in m.get(item) and stack and item in m.get(stack[-1]):
                    stack.append(item)
                    idc = idc + [idx, idx] if item == -b else idc + [idx]
                    if len(idc) == 2:
                        print()
                    marks.append(Mark(idc, coords[idc]))
                    stack = []
                    idc = []
                    continue
            except IndexError:
                print()
            if (not stack and item in (-a, b)) or (stack and item in m.get(stack[-1], [])):
                stack.append(item)
                idc = idc + [idx, idx] if item == b else idc + [idx]
            else:
                stack = []
                idc = []
        return marks

    marks = process_sequence(make_sequence(coords, height, height_tol), coords)
    checkpoint = Checkpoint(marks) if marks else False
    if checkpoint and full_check(checkpoint, gaps, width, n, gap_tol, width_tol):
        return True, checkpoint
    else:
        return False, None
