import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class Kmeans():
    """
    Parameters:
    k: int
        The number of iterations the alogorithm will form.
    max_iterations: int
        The number of iterations the algorithm with run for it if it does
        not converge before that.
    """

    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def  _init_random_centroids(self, X):
        """ Initialize the centroids as k random samples of X"""
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))  # Centroids with features value
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def _euclidean_distance(self, x, y):
        """
        x = vector 1
        y = vector 2
        """
        return np.sqrt(np.sum((x-y)**2))

    def _closest_centroid(self, sample, centroids):
        """ Return the index of the closest centroid to the sample"""
        def _euclidean_distance(x, y):
            """
            x = vector 1
            y = vector 2
            """
            return np.sqrt(np.sum((x-y)**2))


        closest_i = 0
        closest_dist = float("inf")
        for i, centroid in enumerate(centroids):
            distance = _euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i
    
    def _create_clusters(self, centroids, X):
        """ Assign the samples to the closest centroids to create clusters
        return clusters with sample_id 
        """
        #n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _calculate_centroids(self, clusters, X):
        """ Calculate new centroids as the means of the samples in each cluster  """
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids
 
    def _get_cluster_labels(self, clusters, X):
        """Classify samples as the index of their clusters """
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        """ Do K-Means clustering and return cluster indices """

        # Initialize centroids as k random samples from X
        centroids = self._init_random_centroids(X)
        print(centroids)

        # Iterate until convergence or for max iterations
        for _ in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, X)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculate_centroids(clusters, X)
            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self._get_cluster_labels(clusters, X)

if __name__ == "__main__":
    X, y = make_blobs(n_samples=800, centers=4, cluster_std=0.7)
    plt.scatter(X[:,0],X[:,1])
    model = Kmeans(k = 4)
    y_pred = model.predict(X)
    plt.scatter(X[:,0],X[:,1],c=y_pred,cmap="viridis")
