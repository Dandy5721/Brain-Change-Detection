import numpy as np
from joblib import Parallel, delayed
from math import inf
from geometry import Euclidean, SPD


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((distance / bandwidth) ** 2)
    )


class PointGrouper:
    def __init__(self, geometry, group_distance_tolerance=0.1):
        self.geometry = geometry
        self.group_distance_tolerance = group_distance_tolerance

    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        cluster_centers = []
        log_centers = []
        log_points = []

        for p in points:
            log_points.append(self.geometry.logm(p))

        for point, log_point in zip(points, log_points):
            nearest_group_index = self._determine_nearest_group(log_point, log_centers)
            if nearest_group_index is None:
                groups.append([point])
                cluster_centers.append(point)
                log_centers.append(log_point)
                group_assignment.append(group_index + 1)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index + 1)
                groups[nearest_group_index].append(point)

        return np.array(group_assignment), np.array(cluster_centers)

    def _determine_nearest_group(self, log_point, log_centers):
        nearest_group_index = None
        min_distance = inf

        for index, log_center in enumerate(log_centers):
            distance_to_group = self.geometry.norm(log_point - log_center)
            if (
                distance_to_group < self.group_distance_tolerance
                and distance_to_group < min_distance
            ):
                nearest_group_index = index
                min_distance = distance_to_group

        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_distance = inf
        for p in group:
            dist = self.geometry.distance(point, p)
            if dist < min_distance:
                min_distance = dist
        return min_distance


class ManifoldMeanShift:
    def __init__(
        self,
        bandwidth=None,
        n_clusters=None,
        geometry='SPD',
        max_iter=200,
        n_jobs=-1,
        max_bandwidth=10,
        estimate_max_iter=30,
    ):
        if geometry == 'Euclidean':
            geometry = Euclidean()
        else:
            geometry = SPD()

        self.kernel = gaussian_kernel
        self.bandwidth = bandwidth
        self.n_clusters = n_clusters
        self.geometry = geometry
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.max_bandwidth = max_bandwidth
        self.estimate_max_iter = estimate_max_iter

    def cluster(self, points, bandwidth):
        log_points = []
        for p in points:
            log_points.append(self.geometry.logm(p))

        shifted_points = Parallel(n_jobs=self.n_jobs)(
            delayed(self._shift_point)(bandwidth, point, points, log_points)
            for point in points
        )

        point_grouper = PointGrouper(self.geometry)
        labels, cluster_centers = point_grouper.group_points(shifted_points)
        return labels, cluster_centers, np.array(shifted_points)

    def fit(self, points):
        if self.bandwidth is None:
            self.estimate_bandwidth(points)
        elif self.bandwidth > 0:
            self.labels, self.cluster_centers, self.shifted_points = self.cluster(
                points, self.bandwidth
            )
        return self

    def _shift_point(self, bandwidth, point, points, log_points):
        geo = self.geometry

        stop_thresh = 0.000001
        max_iter = self.max_iter
        iter = 1
        ms_norms = []

        while True:
            sum_weights = 0
            ms_vector = np.zeros_like(point)
            log_y = geo.logm(point)
            for (p, log_x) in zip(points, log_points):
                distance = geo.norm(log_x - log_y)
                weight = self.kernel(distance, bandwidth)
                delta = log_x - log_y
                ms_vector += weight * delta
                sum_weights += weight

            ms_vector = ms_vector / sum_weights

            point = geo.expmm(point, ms_vector)
            ms_norm = geo.norm(ms_vector)
            ms_norms.append(ms_norm)

            if ms_norm < stop_thresh or iter >= max_iter:
                break
            iter += 1

        return point

    def estimate_bandwidth(self, points):
        iter = 0
        left, right = 0, self.max_bandwidth
        while right >= left and iter < self.estimate_max_iter:
            iter += 1
            mid = (left + right) / 2
            labels, cluster_centers, shifted_points = self.cluster(points, mid)
            n_clusters = len(cluster_centers)

            if n_clusters == self.n_clusters:
                break
            elif n_clusters > self.n_clusters:
                left = mid
            else:
                right = mid

        self.labels, self.cluster_centers, self.shifted_points = (
            labels,
            cluster_centers,
            shifted_points,
        )

        return mid
