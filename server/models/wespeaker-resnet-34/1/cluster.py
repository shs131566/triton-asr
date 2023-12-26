import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist


class AgglomerativeClustering:
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 1,
        max_clusters: int = 20,
        num_clusters: int = None,
    ):
        num_embeddings = embeddings.shape[0]
        min_cluster_size = 1
        with np.errstate(divide="ignore", invalid="ignore"):
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        dendrogram: np.ndarray = linkage(
            embeddings, method="centroid", metric="euclidean"
        )

        clusters = fcluster(dendrogram, 0.9, criterion="distance") - 1
        cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
        print(cluster_unique, cluster_counts)
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        if num_large_clusters < min_clusters:
            num_clusters = min_clusters
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        if num_clusters is not None and num_large_clusters != num_clusters:
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.threshold)):
                new_cluster_size = _dendrogram[iteration, 3]
                if new_cluster_size < min_cluster_size:
                    continue

                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)

                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                if num_large_clusters == num_clusters:
                    break
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )
        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters
