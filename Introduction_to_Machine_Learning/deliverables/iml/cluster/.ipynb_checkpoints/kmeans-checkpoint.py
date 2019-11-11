import numpy as np


class KMeans:
    """
        K-means
    """
    def __init__(self, n_clusters=3, random_state=12):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X):
        #Randomly initialize centroids
        random_state = np.random.RandomState(self.random_state)
        centroids = X[random_state.choice(X.shape[0], self.n_clusters)]
        centroids_updated = centroids.copy()

        distances = np.zeros((X.shape[0], self.n_clusters)) # Errors

        while True:

            #Cluster assignment step
            for index in range(self.n_clusters):
                distances[:, index] = np.linalg.norm(X-centroids[index], axis=1)

            labels = np.argmin(distances, axis=1)

            # Move centroid step
            centroids_updated = np.array([X[labels == index].mean(axis=0)
                                          for index in range(self.n_clusters)])

            # If centroids don't move
            if np.all(centroids == centroids_updated):
                break

            centroids = centroids_updated

        # Useful for elbow method. Sum of errors.
        inertia = sum(np.min(distances, axis=1)) #/ X.shape[0]

        self.centroids = centroids
        self.labels = labels
        self.inertia = inertia

        return self
