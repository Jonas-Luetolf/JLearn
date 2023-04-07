import numpy as np


class BaseClassifier:
    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self, X_train, y_train) -> None:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """
        predicts the class of the given points

        :param X: a array with all points to classify
        :return : a array with all classes of the points
        """

        return np.array([self._predict(x) for x in X])

    def _predict(self, x) -> np.ndarray:
        """
        classification of one point

        :param x: a array with all features of the point
        :return : the class of the point
        """
        raise NotImplementedError


class BaseRegression:
    def __init__(self, lr=0.001, n_iters=1000) -> None:
        """
        :param lr: the learning rate for gradient descent
        :param n_iters: number of gradient descent steps
        """

        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray
        self.bias: float

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        trains the model with gradient descent
        :param X_train: the training features
        :param y_train: the true values
        :return : None
        """

        n_samples, n_features = X_train.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = self._calculate(X_train)

            dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (1 / n_samples) * np.sum(y_predicted - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def _calculate(self, X) -> np.ndarray:
        """
        calculates the prediction for a dataset
        :param X: the dataset to predict
        :return : predicted values
        """
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """
        returns the predicted values or class
        :param X: dataset to predict
        :return : predicted values or class
        """

        return self._calculate(X)
