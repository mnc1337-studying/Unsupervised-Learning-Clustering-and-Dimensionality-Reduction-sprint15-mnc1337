import numpy as np
import matplotlib.pyplot as plt

# K-Means Clustering Implementation
class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.labels = []
        self.random_state = random_state

    def fit(self, data, initial_centroids=None):
        data = np.array(data)

        # 1. Initialize centroids
        if initial_centroids:
            self.centroids = np.array(initial_centroids)
        else:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(data), self.k, replace=False)
            self.centroids = data[indices]

        for iteration in range(self.max_iters):
            # 2. Assign labels
            labels = []
            for point in data:
                distances = [self.euclidean_distance(point, c) for c in self.centroids]
                labels.append(np.argmin(distances))
            labels = np.array(labels)

            # 3. Compute new centroids
            new_centroids = []
            for i in range(self.k):
                cluster_points = data[labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    # If a cluster has no points, keep old centroid
                    new_centroids.append(self.centroids[i])
            new_centroids = np.array(new_centroids)

            # 4. Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels = labels.tolist()
        return self

    def predict(self, point):
        distances = [self.euclidean_distance(point, c) for c in self.centroids]
        return np.argmin(distances)

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


# Visualization function
def plot_clusters(data, labels, centroids):
    data = np.array(data)
    labels = np.array(labels)
    centroids = np.array(centroids)

    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Annual Income ($1000)')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()


# Main Execution
if __name__ == "__main__":
    data = [
        [15, 39], [15, 81], [16, 6], [16, 77], [17, 40],
        [18, 6], [18, 94], [19, 3], [19, 72], [20, 44]
    ]

    kmeans = KMeans(k=3)
    kmeans.fit(data)

    print("Centroids:", kmeans.centroids)
    print("Labels:", kmeans.labels)
    plot_clusters(data, kmeans.labels, kmeans.centroids)