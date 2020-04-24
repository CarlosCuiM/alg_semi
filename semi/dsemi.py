# -*- coding: UTF-8 -*-

# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from .kernel import RbfKernel
from .loss import HingeLoss
from .manifold_reg import RbfManifold

__all__ = ["Nsemi"]


class Nsemi:

    def __init__(self, kernel=RbfKernel(), loss=HingeLoss(),
                 num_support_vector=0, support_vectors=[],
                 sample_weight=None, manifold=RbfManifold()):

        self._num_support_vectors = num_support_vector
        if len(support_vectors) != 0:
            self._support_vectors = support_vectors
        else:
            self._support_vectors = []
        self._kernel = kernel
        self._loss = loss
        self._num_error = 0
        self._accuracy = []
        self._pred_collection = []
        self._current_step = 0
        self._sample_weight = sample_weight
        self._num_observations = 0
        self._confusion_matrix = []
        self._manifold = manifold

    def _predict(self, x):
        return np.dot(self._sample_weight, x)

    def train(self, X, y, learning_rate=0.1, reg_coefficient=0.0):
        int_set = ["int16", "int32"]
        assert learning_rate > 0, "Error:leaning rate must be positive."
        assert reg_coefficient >= 0, "regularization coefficient must be non-negative."

        self._num_observations += len(y)

        self._confusion_matrix = np.zeros((2, 2))
        for t in range(self._num_observations):

            self._update(X[t], y[t], learning_rate, reg_coefficient)

    def _update(self, x, y, learning_rate, reg_coefficient):
        if self._num_support_vectors > 0:
            kernel_vector = self._kernel.compute_kernel(np.array(self._support_vectors), x)
            pred_value = self._predict(kernel_vector)
        else:
            pred_value = 0

        pred_label = 1 if pred_value > 0 else -1

        self._pred_collection.append(pred_label)
        true_label = y

        if true_label != 0:
            loss = None
            self._count_num_error(pred_label, true_label)
            loss = self._loss.loss_computing(pred_value, true_label)
            self._accuracy.append(self._get_accuracy(self._current_step + 1))
            self._current_step += 1

            if loss > 0:
                gradient = None
                if isinstance(self._sample_weight, np.ndarray):
                    self._sample_weight = np.hstack((self._sample_weight, np.zeros((1,))))
                else:
                    self._sample_weight = np.zeros((1,))
                gradient_absolute = self._kernel.compute_kernel(x,x)
                gradient = -gradient_absolute
                self._sample_weight[self._num_support_vectors] = -learning_rate * true_label * gradient
                self._support_vectors.append(x)
                self._num_support_vectors += 1

            else:
                if self._num_support_vectors > 1 and reg_coefficient != 0:
                    self._sample_weight[self._num_support_vectors - 1] = (1 - learning_rate * reg_coefficient) * \
                        self._sample_weight[self._num_support_vectors - 1]

    def _get_accuracy(self, t):
        return 1 - self._num_error / t

    def plot_accuracy_curve(self, x_label="Number of sample", y_label='Accuracy', title=None):

        print(self._accuracy)

        plt.plot(self._accuracy)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def plot_confusion_matrix(self, x_label="True label", y_label="Prediction", title=None):

        print(self._confusion_matrix)

        plt.imshow(self._confusion_matrix)
        ticks = range(2)
        plt.xticks(ticks, [-1, 1, ])
        plt.yticks(ticks, [-1, 1, ])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        loc = 0
        if np.any(self._confusion_matrix - 100 >= 0):
            loc = 0.25
        for x_tick in ticks:
            for y_tick in ticks:
                plt.text(x_tick - loc, y_tick, int(self._confusion_matrix[x_tick, y_tick]), color="white")
        plt.show()

    def get_accuracy(self):
        return self._accuracy

    def _count_num_error(self, pred_label, true_label):
        if pred_label == -1:
            pred_label = 0
        if true_label == -1:
            true_label = 0
        if pred_label != true_label:
            self._num_error += 1
            self._confusion_matrix[true_label, pred_label] += 1
        else:
            self._confusion_matrix[true_label, true_label] += 1
