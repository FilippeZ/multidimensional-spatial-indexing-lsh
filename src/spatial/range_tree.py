"""
Multidimensional Range Tree implementation.
Supports logarithmic range queries and kNN search across numerical spatial dimensions.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import heapq


class RangeTreeNode:
    def __init__(
        self,
        point: Tuple[float, ...],
        axis: int,
        left: Optional["RangeTreeNode"] = None,
        right: Optional["RangeTreeNode"] = None,
        subtree: Optional["RangeTree"] = None,
        bounds_min: Optional[Tuple[float, ...]] = None,
        bounds_max: Optional[Tuple[float, ...]] = None,
    ):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        self.subtree = subtree
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max


class RangeTree:
    def __init__(self, points: Optional[List[Tuple[float, ...]]] = None, axis: int = 0):
        self.points: List[Tuple[float, ...]] = points[:] if points else []
        self.root: Optional[RangeTreeNode] = None
        self.axis = axis
        self.k = len(self.points[0]) if self.points else 0

        if self.points:
            self.build(self.points)

    def build(self, points: List[Tuple[float, ...]]) -> None:
        self.points = points[:]
        if not points:
            self.root = None
            self.k = 0
            return

        self.k = len(points[0])
        axis = self.axis
        pts_sorted = sorted(points, key=lambda p: p[axis])

        def compute_bounds(pts: List[Tuple[float, ...]]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
            mins = [min(p[d] for p in pts) for d in range(self.k)]
            maxs = [max(p[d] for p in pts) for d in range(self.k)]
            return tuple(mins), tuple(maxs)

        def build_rec(pts: List[Tuple[float, ...]], axis: int) -> Optional[RangeTreeNode]:
            if not pts:
                return None

            mid = len(pts) // 2
            median_point = pts[mid]

            left_node = build_rec(pts[:mid], axis)
            right_node = build_rec(pts[mid + 1:], axis)

            bounds_min, bounds_max = compute_bounds(pts)

            if axis + 1 < self.k:
                subtree = RangeTree(pts, axis=axis + 1)
            else:
                subtree = None

            return RangeTreeNode(
                point=median_point,
                axis=axis,
                left=left_node,
                right=right_node,
                subtree=subtree,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )

        self.root = build_rec(pts_sorted, axis)

    def insert(self, point: Tuple[float, ...]) -> None:
        if self.k and len(point) != self.k:
            raise ValueError(f"Point dimensionality {len(point)} != tree dimensionality {self.k}")
        self.points.append(point)
        self.build(self.points)

    def delete(self, point: Tuple[float, ...]) -> None:
        if point in self.points:
            self.points.remove(point)
            self.build(self.points)

    def range_query(self, query_ranges: List[Tuple[float, float]]) -> List[Tuple[float, ...]]:
        if self.k == 0:
            return []

        if len(query_ranges) != self.k:
            raise ValueError("Length of query_ranges must equal dimensionality of points (k).")

        results: List[Tuple[float, ...]] = []

        def rect_outside(bounds_min: Tuple[float, ...], bounds_max: Tuple[float, ...]) -> bool:
            for d in range(self.k):
                if bounds_max[d] < query_ranges[d][0] or bounds_min[d] > query_ranges[d][1]:
                    return True
            return False

        def rect_inside(bounds_min: Tuple[float, ...], bounds_max: Tuple[float, ...]) -> bool:
            for d in range(self.k):
                if not (query_ranges[d][0] <= bounds_min[d] and bounds_max[d] <= query_ranges[d][1]):
                    return False
            return True

        def point_in_all_dims(p: Tuple[float, ...]) -> bool:
            return all(query_ranges[d][0] <= p[d] <= query_ranges[d][1] for d in range(self.k))

        def collect_all(node: Optional[RangeTreeNode]) -> None:
            if node is None:
                return
            collect_all(node.left)
            if point_in_all_dims(node.point):
                results.append(node.point)
            collect_all(node.right)

        def query_rec(node: Optional[RangeTreeNode]) -> None:
            if node is None:
                return
            bmin, bmax = node.bounds_min, node.bounds_max
            
            if rect_outside(bmin, bmax):
                return

            if rect_inside(bmin, bmax):
                if node.subtree:
                    results.extend(node.subtree.range_query(query_ranges))
                else:
                    collect_all(node)
                return

            if point_in_all_dims(node.point):
                results.append(node.point)

            query_rec(node.left)
            query_rec(node.right)

        query_rec(self.root)

        if not results:
            return []
        seen = set()
        unique = []
        for p in results:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def kNN(self, target: Tuple[float, ...], k: int) -> List[Tuple[float, ...]]:
        if self.k == 0 or len(target) != self.k or k <= 0:
            return []

        heap: List[Tuple[float, Tuple[float, ...]]] = []

        def push_candidate(p: Tuple[float, ...]) -> None:
            d2 = sum((p[d] - target[d])**2 for d in range(self.k))
            item = (-d2, p)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif d2 < -heap[0][0]:
                heapq.heapreplace(heap, item)

        def bbox_min_dist_sq(bmin: Tuple[float, ...], bmax: Tuple[float, ...]) -> float:
            acc = 0.0
            for d in range(self.k):
                if target[d] < bmin[d]:
                    acc += (bmin[d] - target[d])**2
                elif target[d] > bmax[d]:
                    acc += (target[d] - bmax[d])**2
            return acc

        def best_radius_sq() -> float:
            return -heap[0][0] if heap else float("inf")

        def search(node: Optional[RangeTreeNode]) -> None:
            if node is None:
                return

            if bbox_min_dist_sq(node.bounds_min, node.bounds_max) >= best_radius_sq():
                return

            push_candidate(node.point)
            axis = node.axis
            go_left_first = target[axis] <= node.point[axis]
            first, second = (node.left, node.right) if go_left_first else (node.right, node.left)

            search(first)
            if second is not None:
                if bbox_min_dist_sq(second.bounds_min, second.bounds_max) < best_radius_sq():
                    search(second)

        search(self.root)
        heap_sorted = sorted(heap, key=lambda t: -t[0])
        return [p for _, p in heap_sorted]
