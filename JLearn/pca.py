import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components: int = n_components
        self.components: np.ndarray

    def fit(self, X: np.ndarray) -> None:
        X -= np.mean(X, axis=0)
        X_cov = np.cov(X, rowvar=False)

        # calculate eigvalues and eigvectors
        eig_vals, eig_vecs = np.linalg.eig(X_cov)
        eig_vecs = eig_vecs.T

        # sort eigvectors
        idxs = np.argsort(eig_vals)[::-1]
        self.components = (eig_vecs[idxs])[: self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X -= np.mean(X, axis=0)
        return np.dot(X, self.components.T)
