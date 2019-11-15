import numpy as np
from scipy import stats


class KModes:
    """
        K-means
    """
    def __init__(self, n_clusters=3, max_iter=100, random_state=12):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _dissimilarities(self,Xs, centroid):
        # if sample == centroid then 0, else 1
        delta = lambda x_i, z_i: int(x_i != z_i)
        result = [sum([delta(x, z) for x, z in zip(X, centroid)]) for X in Xs]
        return np.array(result)

    def fit(self, X):
        # Randomly initialize centroids
        random_state = np.random.RandomState(self.random_state)
        centroids = X[random_state.choice(X.shape[0], self.n_clusters)]
        centroids_updated = centroids.copy()

        # Update the centroid with ones with most coincidence (more zeros than ones)
        min_mode = lambda x: [stats.mode(x[:, i])[0][0] for i in range(len(x[0, :]))]
        dissimilarities = np.zeros((X.shape[0], self.n_clusters))

        labels = None
        for iter in range(self.max_iter):

            # Cluster assignment step
            for index in range(self.n_clusters):
                # distances[:, index] = np.linalg.norm(X-centroids[index], axis=1)
                dissimilarities[:, index] = self._dissimilarities(X, centroids[index])

            labels = np.argmin(dissimilarities, axis=1)
            # Move centroid step
            try:
                centroids_updated = np.array([min_mode(X[labels == index]) for index in range(self.n_clusters)])
            except:
                pass

            # If centroids don't move
            # print(f'iteration: {iter}')
            if np.all(centroids == centroids_updated):
                # print('Converged.')
                break

            centroids = centroids_updated

        # Useful for elbow method. Sum of errors.
        inertia = sum(np.min(dissimilarities, axis=1)) #/ X.shape[0]

        self.centroids = centroids
        self.labels = labels
        self.inertia = inertia

        return self
