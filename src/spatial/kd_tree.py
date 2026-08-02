"""
KD-Tree implementation for multi-dimensional spatial indexing.
Provides logarithmic search complexity O(log N) for spatial range and kNN queries.
"""

from typing import List, Tuple, Optional
import heapq


class KDTreeNode:
    def __init__(self, point: Tuple[float, ...], axis: int, left=None, right=None, depth=0):
        self.point = point               # Coordinates of point (k-dimensional tuple)
        self.axis = axis                 # Split axis for this depth level
        self.left = left                 # Left subtree
        self.right = right               # Right subtree
        self.depth = depth               # Node depth level


class KDTree:
    def __init__(self, points: Optional[List[Tuple[float, ...]]] = None):
        self.k = len(points[0]) if points else 0
        self.root = self.build(points) if points else None

    def build(self, points: List[Tuple[float, ...]], depth: int = 0) -> Optional[KDTreeNode]:
        if not points:
            return None
        axis = depth % self.k
        sorted_points = sorted(points, key=lambda x: x[axis])
        median = len(sorted_points) // 2
        return KDTreeNode(
            point=sorted_points[median],
            axis=axis,
            left=self.build(sorted_points[:median], depth + 1),
            right=self.build(sorted_points[median + 1:], depth + 1),
            depth=depth
        )

    def insert_point(self, node: Optional[KDTreeNode], point: Tuple[float, ...], depth: int = 0) -> KDTreeNode:
        if node is None:
            return KDTreeNode(point, axis=depth % self.k, depth=depth)
        axis = node.axis
        if point[axis] < node.point[axis]:
            node.left = self.insert_point(node.left, point, depth + 1)
        else:
            node.right = self.insert_point(node.right, point, depth + 1)
        return node

    def insert(self, point: Tuple[float, ...]):
        if self.k == 0:
            self.k = len(point)
        self.root = self.insert_point(self.root, point)

    def distance_sq(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        return sum((a[i] - b[i]) ** 2 for i in range(len(a)))

    def kNN(self, target: Tuple[float, ...], k: int) -> List[Tuple[float, ...]]:
        heap = []

        def recursive_search(node: Optional[KDTreeNode]):
            if node is None:
                return
            point = node.point
            dist = self.distance_sq(point, target)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, point))
            else:
                if dist < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist, point))

            axis = node.axis
            diff = target[axis] - point[axis]
            close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)
            recursive_search(close)

            if len(heap) < k or diff ** 2 < -heap[0][0]:
                recursive_search(away)

        recursive_search(self.root)
        return [p for (_, p) in sorted(heap, reverse=True)]

    def range_query(self, query_ranges: List[Tuple[float, float]]) -> List[Tuple[float, ...]]:
        results = []

        def is_point_in_range(point):
            return all(query_ranges[i][0] <= point[i] <= query_ranges[i][1] for i in range(self.k))

        def is_region_contained(region):
            for i in range(self.k):
                if region[i][0] < query_ranges[i][0] or region[i][1] > query_ranges[i][1]:
                    return False
            return True

        def regions_intersect(region):
            for i in range(self.k):
                if region[i][1] < query_ranges[i][0] or region[i][0] > query_ranges[i][1]:
                    return False
            return True

        def add_subtree(node):
            if node is None:
                return
            results.append(node.point)
            add_subtree(node.left)
            add_subtree(node.right)

        def recursive_range(node, region):
            if node is None:
                return

            if is_region_contained(region):
                add_subtree(node)
                return

            if is_point_in_range(node.point):
                results.append(node.point)
            axis = node.axis
            point_value = node.point[axis]

            if node.left:
                left_region = [list(r) for r in region]
                left_region[axis][1] = point_value
                if regions_intersect(left_region):
                    recursive_range(node.left, left_region)

            if node.right:
                right_region = [list(r) for r in region]
                right_region[axis][0] = point_value
                if regions_intersect(right_region):
                    recursive_range(node.right, right_region)

        initial_region = [[float('-inf'), float('inf')] for _ in range(self.k)]
        recursive_range(self.root, initial_region)
        return results
