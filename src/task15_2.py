import numpy as np
from collections import deque

# DBSCAN Clustering Implementation
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []
        self.noise_label = -1

    def fit(self, data):
        data = np.array(data)
        n_points = len(data)
        self.labels = [None] * n_points
        cluster_id = 0

        for i in range(n_points):
            if self.labels[i] is not None:
                continue

            neighbors = self.region_query(data, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = self.noise_label
            else:
                self.expand_cluster(data, i, neighbors, cluster_id)
                cluster_id += 1

        return self

    def expand_cluster(self, data, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            idx = queue.popleft()
            if self.labels[idx] == self.noise_label:
                self.labels[idx] = cluster_id
            if self.labels[idx] is not None:
                continue

            self.labels[idx] = cluster_id
            idx_neighbors = self.region_query(data, idx)
            if len(idx_neighbors) >= self.min_samples:
                queue.extend(idx_neighbors)

    def region_query(self, data, point_idx):
        neighbors = []
        for i, point in enumerate(data):
            if self.euclidean_distance(data[point_idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


# Visualization function
def plot_clusters(data, labels):
    import matplotlib.pyplot as plt
    data = np.array(data)
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            color = "red"
            label_name = "Noise"
        else:
            color = np.random.rand(3,)
            label_name = f"Cluster {label}"

        cluster_points = data[np.array(labels) == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=label_name)

    plt.title("DBSCAN Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


# Main Execution
if __name__ == "__main__":
    data = [
        [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]
    ]
    eps = 2
    min_samples = 2

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    print("Labels:", dbscan.labels)
    plot_clusters(data, dbscan.labels)
