# -*- coding: UTF-8 -*-
from abc import abstractmethod, ABC

import numpy as np

_all__ = ["RbfManifold"]


class Manifold(ABC):
    @abstractmethod
    def compute_weight(self, support, x):
        raise NotImplementedError("this is an abstract class.")


class RbfManifold(Manifold):

    def __init__(self, d=1):
        self._dimension = None
        self._precision = d

    def compute_weight(self, support_vector, x):
        shift = support_vector - x
        if np.ndim(support_vector) != 1:
            return np.exp(-self._precision * np.linalg.norm(shift, axis=1))
        else:
            return np.exp(-self._precision * np.dot(shift, shift))


if __name__ == '__main__':
    x_1 = np.array([[1, 2, 3, 2, 4], [3, 4, 6, 2, 1], [3, 4, 6, 2, 1]])
    x_2 = np.array([2, 3, 4, 6, 9])

    manifold = RbfManifold()
    weight = manifold.compute_weight(x_1, x_2)
    print(weight)
