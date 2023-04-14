from ._base import BaseRegression
import numpy as np


class LinearRegression(BaseRegression):
    def _calculate(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression(BaseRegression):
    def _calculate(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        the sigmoid function to classifiy the values
        """
        return 1 / (np.exp(-X) + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        classifiys the data with the sigmoid function

        :param X: dataset to classifiy
        :return : classes of the dataset
        """

        return np.array([1 if i > 0.5 else 0 for i in super().predict(X)])
