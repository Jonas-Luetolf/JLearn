from collections import Counter
import numpy as np


def entropy(y) -> np.ndarray:
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, value=None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left = left
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples=2, max_depth=100) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Node

    def fit(self, X, y) -> None:
        self.n_feats = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples:
            return Node(value=self._most_common(y))

        featidxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feature, best_thresh = self._best_criteria(X, y, featidxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _best_criteria(self, X, y, featidxs) -> tuple:
        best_gain = np.array(-1)
        split = (None, None)

        for featidx in featidxs:
            X_column = X[:, featidx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._info_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split = (featidx, threshold)
        return split

    def _info_gain(self, y, X_column, threshold) -> np.ndarray:
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return np.array(0)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh) -> tuple:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common(self, y) -> np.ndarray:
        c = Counter(y)
        return c.most_common(1)[0][0]

    def predict(self, X) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node) -> np.ndarray:
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
