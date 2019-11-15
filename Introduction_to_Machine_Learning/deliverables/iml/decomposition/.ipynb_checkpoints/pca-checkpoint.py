import numpy as np

class PCA:
    """
        PCA
    """

    def __init__(self, n_components=3, random_state=12):
        self.n_components = n_components
        self.random_state = random_state
        self.W = None

    def fit(self, X, y=None):
        self._fit(X)

        return self

    def _fit(self, X, y=None):
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigen_sorted = sorted(zip(eigenvalues, eigenvectors),
                              key=lambda x: x[0], reverse=True)
        self.W = np.hstack(tuple(eigen[1][:, np.newaxis]
                                 for eigen in eigen_sorted[:self.n_components]))

        total_evalues = sum(eigenvalues)
        self.explained_variance_ratio_ = [(eigen[0]/total_evalues)
                                           for eigen in eigen_sorted[:self.n_components]]
        return self.W

    def fit_transform(self, X, y=None):
        W = self._fit(X)

        return X.dot(W)
