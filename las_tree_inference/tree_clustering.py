"""Tree-top detection and constrained XY clustering for LAS point clouds.

This module contains no file I/O or plotting so it can be imported safely.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

UNCLASSIFIED = -1


def find_possible_tree_tops(points: np.ndarray, height_threshold: float, search_radius: float) -> list[np.ndarray]:
    """Return local XY maxima whose Z value exceeds ``height_threshold``."""
    tree = cKDTree(points[:, :2])
    candidates = []
    for index, point in enumerate(points):
        if point[2] <= height_threshold:
            continue
        neighbors = tree.query_ball_point(point[:2], search_radius)
        if all(points[neighbor][2] <= point[2] for neighbor in neighbors if neighbor != index):
            candidates.append(point)
    return candidates


def confirm_tree_tops(candidates: list[np.ndarray], minimum_spacing: float) -> list[np.ndarray]:
    """Keep only the highest candidate in each XY neighborhood."""
    if not candidates:
        return []

    candidate_array = np.asarray(candidates)
    tree = cKDTree(candidate_array[:, :2])
    accepted = []
    for index in np.argsort(candidate_array[:, 2])[::-1]:
        point = candidate_array[index]
        neighbors = tree.query_ball_point(point[:2], minimum_spacing)
        if all(candidate_array[neighbor][2] <= point[2] for neighbor in neighbors):
            accepted.append(point)
    return accepted


def _nearest_top_distance(point: np.ndarray, tops: list[np.ndarray]) -> float:
    distances = [np.linalg.norm(point[:2] - other[:2]) for other in tops if not np.array_equal(point, other)]
    return min(distances) if distances else float("inf")


def _downhill_neighbors(tree: cKDTree, points: np.ndarray, index: int, epsilon: float) -> list[int]:
    point = points[index]
    neighbors = tree.query_ball_point(point[:2], epsilon)
    return [neighbor for neighbor in neighbors if points[neighbor][2] <= point[2]]


def cluster_from_tree_tops(
    points: np.ndarray,
    tops: list[np.ndarray],
    epsilon: float,
    minimum_spacing: float,
    min_neighbors: int = 1,
) -> np.ndarray:
    """Assign points to top-seeded, downhill-growing clusters.

    Points in a protected inner radius belong to one tree only. Points in the
    outer radius can extend the search but are not assigned, which prevents
    adjacent crowns from merging through their edges.
    """
    labels = np.full(len(points), UNCLASSIFIED, dtype=np.int64)
    if not tops:
        return labels

    tree = cKDTree(points[:, :2])
    top_tree = cKDTree(np.asarray(tops)[:, :2])

    for cluster_id, top in enumerate(tops):
        start_index = int(top_tree.query(top[:2])[1])
        top_point = tops[start_index]
        matches = np.flatnonzero(np.all(np.isclose(points, top_point), axis=1))
        if len(matches) == 0:
            continue
        start_index = int(matches[0])
        if labels[start_index] != UNCLASSIFIED:
            continue

        nearest_distance = _nearest_top_distance(top_point, tops)
        inner_radius = minimum_spacing if not np.isfinite(nearest_distance) else min(minimum_spacing, nearest_distance / 2)
        outer_radius = maximum_radius = max(inner_radius, nearest_distance - minimum_spacing) if np.isfinite(nearest_distance) else minimum_spacing * 2

        labels[start_index] = cluster_id
        visited = {start_index}
        seeds = _downhill_neighbors(tree, points, start_index, epsilon)
        cursor = 0
        while cursor < len(seeds):
            seed_index = seeds[cursor]
            cursor += 1
            if seed_index in visited:
                continue
            visited.add(seed_index)
            distance = np.linalg.norm(points[seed_index, :2] - top_point[:2])
            if distance > outer_radius:
                continue

            neighbors = _downhill_neighbors(tree, points, seed_index, epsilon)
            if len(neighbors) < min_neighbors:
                continue
            if distance <= inner_radius and labels[seed_index] == UNCLASSIFIED:
                labels[seed_index] = cluster_id
            for neighbor in neighbors:
                if neighbor not in visited:
                    seeds.append(neighbor)

    return labels
