from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from ezdxf.math.vector import Vector

@dataclass
class Mark:
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
    marks: Tuple[Mark]
    n: int = field(init=False)
    gaps: Tuple = field(init=False)

    def __post_init__(self):
        self.n = len(self.marks)
        self.gaps = tuple(
            self.marks[i - 1].coords[-1].xy.distance(self.marks[i].coords[0].xy) for i in range(1, self.n))


def checker(coords, height, width=None, gaps=None, n: int = None, tol: float = 0.5, *, a: int = 1, b: int = 3,
            **kwargs) -> (bool, Checkpoint):
    # TODO: переписать как класс
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
        sequence = np.copy(coords[..., Z])
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
