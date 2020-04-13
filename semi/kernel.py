# -*- coding: UTF-8 -*-

import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):
    @abstractmethod
    def compute_kernel(self, x_train, x_new):
        raise NotImplementedError("this is an abstract class.")

    @abstractmethod
    def get_dimension(self):
        raise NotImplementedError("this is an abstract class.")


class RbfKernel(Kernel):
    def __init__(self, d=1):
        self._dimension = None
        self._precision = d

    def compute_kernel(self, x_train, x_new):
        shift = x_train - x_new
        if np.ndim(x_train) != 1:
            return np.exp(-self._precision * np.linalg.norm(shift, axis=1))
        else:
            return np.exp(-self._precision * np.dot(shift, shift))

    def get_dimension(self):
        return np.inf


if __name__ == "__main__":
    x_1 = np.array([[1, 2, 3, 2, 4], [3, 4, 6, 2, 1], [3, 4, 6, 2, 1]])
    x_2 = np.array([2, 3, 4, 6, 9])
    print(x_1)
    print(x_2)
    RbfKernel = RbfKernel()
    tt = RbfKernel.compute_kernel(x_1, x_2)
    print(tt)
    print(np.argmax(tt))