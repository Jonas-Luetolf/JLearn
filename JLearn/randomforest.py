from collections import Counter
import numpy as np
from decisiontree import DecisionTree


def get_sample(X: np.ndarray, y: np.ndarray) -> tuple:
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common(preds: np.ndarray) -> np.ndarray:
    c = Counter(preds)
    return c.most_common(1)[0][0]


class RandomForest:
    def __init__(
        self, n_trees: int, min_samples: int = 2, max_depth: int = 100
    ) -> None:
        self.n_trees = n_trees
        self.min_samples = min_samples
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)

            X_samples, y_samples = get_sample(X, y)
            tree.fit(X_samples, y_samples)

            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)

        return np.array([most_common(prediction) for prediction in predictions])
