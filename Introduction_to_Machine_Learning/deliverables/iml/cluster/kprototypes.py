import numpy as np
from scipy import stats


class KPrototypes:
    """
        K-means
    """
    def __init__(self, n_clusters=3, cat_features=[],
                 gamma=0.8, max_iter=100, random_state=12):
        self.n_clusters = n_clusters
        self.cat_features = cat_features
        self.gamma = gamma
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

        # Split
        X_num = np.asanyarray(X[:, [feat for feat in range(X.shape[1])
                                    if feat not in self.cat_features]]).astype(np.float64)
        X_cat = np.asanyarray(X[:, self.cat_features])

        # Paper Huang [1997]
        if self.gamma is None:
            gamma = 0.5 * X_num.std()

        # kmeans
        centroids_kmeans = X_num[random_state.choice(X_num.shape[0], self.n_clusters)]
        centroids_kmeans_updated = centroids_kmeans.copy()
        distances = np.zeros((X_num.shape[0], self.n_clusters))  # Errors

        # kmodes
        centroids_kmodes = X_cat[random_state.choice(X_cat.shape[0], self.n_clusters)]
        centroids_kmodes_updated = centroids_kmodes.copy()
        dissimilarities = np.zeros((X_cat.shape[0], self.n_clusters))

        # kprototypes
        centroids_total = np.concatenate([centroids_kmeans, centroids_kmodes], axis=1)
        centroids_total_updated = centroids_total.copy()
        distances_total = np.zeros((X.shape[0], self.n_clusters))  # Errors

        # Update the centroid with ones with most coincidence (more zeros than ones)
        min_mode = lambda x: [stats.mode(x[:, i])[0][0] for i in range(len(x[0, :]))]

        labels = None

        for iter in range(self.max_iter):

            # Cluster assignment step
            for index in range(self.n_clusters):
                distances[:, index] = np.linalg.norm(X_num - centroids_kmeans[index], axis=1)
                dissimilarities[:, index] = self._dissimilarities(X_cat, centroids_kmodes[index])

            distances_total = dissimilarities + self.gamma * distances

            #     clusters = np.argmin(_dissimilarities, axis=1)
            labels = np.argmin(distances_total, axis=1)

            centroids_kmeans_updated = np.array([X_num[labels == index].mean(axis=0)
                                                 for index in range(self.n_clusters)])
            centroids_kmodes_updated = np.array([min_mode(X_cat[labels == index]) for index in range(self.n_clusters)])

            centroids_total_updated = np.concatenate([centroids_kmeans_updated, centroids_kmodes_updated], axis=1)

            # print(f'iteration: {iter}')
            # If centroids don't move
            if np.all(centroids_total == centroids_total_updated):
                # print('Converged.')
                break

            centroids_total = centroids_total_updated

        # Useful for elbow method. Sum of errors.
        inertia = sum(np.min(distances_total, axis=1)) #/ X.shape[0]

        self.centroids = centroids_total
        self.labels = labels
        self.inertia = inertia

        return self
