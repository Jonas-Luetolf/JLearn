from ._base import BaseClassifier
import numpy as np


class SVM(BaseClassifier):
    def __init__(
        self, lr: float = 0.001, lambda_param: float = 0.01, n__iters: int = 1000
    ) -> None:
        self.lr = lr
        self.lambda_param = lambda_param
        self.n__iters = n__iters

        self.w: np.ndarray
        self.b: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.w = np.zeros(X.shape[1])
        self.b = np.array(0, dtype=np.float128)

        y = np.where(y <= 0, -1, 1)
        print(y)

        for _ in range(self.n__iters):
            for idx in range(len(y)):
                if y[idx] * (np.dot(self.w, X[idx])) - self.b >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w) - np.dot(
                        X[idx], y[idx]
                    )
                    self.b -= self.lr * y[idx]

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(x, self.w) - self.b)
