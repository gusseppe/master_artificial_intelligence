import numpy as np
import copy
import math

from scipy.spatial.distance import cdist # Compute distance between each pair of the two collections of input


class FuzzyCMeans2:
    """ Fuzzy C means own algorithm """

    def __init__(self, n_clusters=3, random_state=0, m=2, max_iter=100, toleration=0.01):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.m = m
        self.max_iter = max_iter
        self.toleration = toleration
        self.U = None
        self.V = None
        self.termination_values = None
        self.iterations = None

    def get_U(self, X, n_clusters, random_state):
        """
        This function initializes the U matrix with random values
        """
        U = []
        np.random.seed(random_state)
        for k in range(0, len(X)):
            U_row = []
            rand_sum = 0.0
            for j in range(0, n_clusters):
                value = np.random.randint(1, int(10000))
                U_row.append(value)
                rand_sum += value
            for j in range(0, n_clusters):
                U_row[j] = U_row[j] / rand_sum
            U.append(U_row)
        return U

    def end_fcm(self, U, U_old, toleration):
        """
        This function ends the fcm algorithm when the difference between two successive U membership functions is less than the toleration value.
        """
        for k in range(0, len(U)):
            for j in range(0, len(U[0])):
                if abs(U[k][j] - U_old[k][j]) > toleration:
                    return False
        return True

    def termination(self, U, U_old, n_clusters):
        """
        This function computes the mean termination value.
        """
        T = []
        for i in range(0, len(U)):
            for j in range(0, n_clusters):
                T.append(np.mean(abs(np.array(U[i]) - np.array(U_old[i]))))

        return np.mean(T)

    def euclidean_distance(self, point, centroid):
        """
        This function euclidean distances bewteen a datapoint and the centroid
        """
        square_diff = 0.0
        for i in range(0, len(point)):
            square_diff += abs(point[i] - centroid[i]) ** 2
        return math.sqrt(square_diff)

    def calculate_cluster_centroids(self, X, U, n_clusters, m):
        V = []
        for j in range(0, n_clusters):
            V_centroid = []
            for i in range(0, len(X[0])):  # number of dimensions (40)
                sum_numerator = 0.0
                sum_denominator = 0.0
                for k in range(0, len(X)):
                    sum_numerator += (U[k][j] ** m) * X[k][i]
                    sum_denominator += (U[k][j] ** m)
                V_centroid.append(sum_numerator / sum_denominator)
            V.append(V_centroid)
        return V

    def update_membership_U(self, X, U, V, n_clusters, m):
        for j in range(0, n_clusters):
            for k in range(0, len(X)):
                # U_old[k][j]= U[k][j]
                sum_value = 0.0
                upper = self.euclidean_distance(X[k], V[j])
                for i in range(0, n_clusters):
                    lower = self.euclidean_distance(X[k], V[i])
                    sum_value += (upper / lower) ** (2 / (m - 1))
                U[k][j] = 1 / sum_value
        return U

    def fit(self, X):
        """
        This is the main function, it would calculate the required center, and return the final normalised membership matrix U.
        It's paramaters are the : cluster number and the fuzzifier "m".
        """
        # initialise the U matrix:
        U = self.get_U(X, self.n_clusters, self.random_state)

        # Initialize the iterations:
        iterations = []
        termination_values = []

        for iteration in range(self.max_iter):
            iterations.append(iteration)
            # create a copy of it, to check the end conditions
            U_old = copy.deepcopy(U)

            # Calculate cluster centroids (V):
            V = self.calculate_cluster_centroids(X, U, self.n_clusters, self.m)

            # update U vector
            U = self.update_membership_U(X, U, V, self.n_clusters, self.m)
            termination_values.append(self.termination(U, U_old, self.n_clusters))
            if self.end_fcm(U, U_old, self.toleration) == True:
                break

        self.U = U
        self.V = V
        self.termination_values = termination_values
        self.iterations = iterations

        return self

class FuzzyCMeans:
    """
        K-means
    """
    def __init__(self, n_clusters=3, m=2, max_iter=100, tol=0.01, random_state=12):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def get_u(self, X, v, m):

        """
        This function Updates the membership function by computing distances between the data points and the centroid matrix v.
        It's paramaters are the : data, centroids matrix and the fuzzifier value"m".

        """

        # compute the distance between dataset an centers by avoiding zeros
        distances = np.fmax(cdist(X, v, metric='euclidean'), np.finfo(np.float64).eps)

        distances2 = distances ** (2 / (m - 1))

        # compute the membership function u
        u = 1 / (distances2.T / np.sum(distances2, axis=1))

        return u

    def fit(self, X):
        # Randomly initialize centroids
        random_state = np.random.RandomState(self.random_state)

        v = X[random_state.choice(X.shape[0], self.n_clusters)]  # k centroids have been generated randomly

        v_updated = v.copy()

        # initialize membership function:
        u = self.get_u(X, v, self.m)

        labels = None

        for iter in range(self.max_iter):

            # old membership function
            u_old = u

            # update centroids
            um = u ** self.m
            v_updated = np.dot(um, X) / np.sum(um, axis=1, keepdims=True)

            # update membership function
            u = self.get_u(X, v, self.m)

            # termination value (difference between membership functions)
            termination = np.linalg.norm(u - u_old)

            # print(f'iteration: {iter}')
            if termination < self.tol:
                # print('Converged.')
                break

            v = v_updated

        # Useful for elbow method. Sum of errors.
        inertia = sum(np.min(u, axis=1)) #/ X.shape[0]

        self.centroids = v
        self.labels = u.argmax(axis=0)
        self.inertia = inertia

        return self
