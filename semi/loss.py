# -*- coding: UTF-8 -*-

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def loss_computing(self, x, **kwargs):
        raise NotImplementedError("this is an abstract method")


class HingeLoss(Loss):

    def loss_computing(self, x, y):
        return max(0, 1 - x * y)


if __name__ == "__main__":
    hl = HingeLoss()

    loss2 = hl.loss_computing(x=2, y=1)
    print(loss2)
