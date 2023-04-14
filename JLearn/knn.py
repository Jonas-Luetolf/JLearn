import numpy as np
from collections import Counter

from ._base import BaseClassifier
from ._distances import euclidean_distance


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors: int) -> None:
        """
        :param n_neighbors: number of neighbors for prediction
        """

        self.n_neighbors = n_neighbors

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        stores the training data

        :param X_train: np array with all training examples
        :param y_train: np array with all labels for the training examples
        """

        self.X_train = X_train
        self.y_train = y_train

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """
        prediction for one point

        :param x: all features of one point
        :return : np array with the label
        """
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_idx = np.argsort(distances)[: self.n_neighbors]
        k_labs = [self.y_train[i] for i in k_idx]

        return Counter(k_labs).most_common(1)[0][0]
