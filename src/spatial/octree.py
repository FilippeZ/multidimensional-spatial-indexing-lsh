"""
Octree implementation for 3D spatial indexing (ZIP, Year, Score).
Efficient for Axis-Aligned Bounding Box (AABB) sub-division and range queries.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import heapq


class OctreeNode:
    def __init__(self, center: List[float], size: float):
        self.center = center
        self.size = float(size)
        self.children: List[Optional["OctreeNode"]] = [None] * 8
        self.is_leaf = True
        self.data: List[dict] = []


class Octree:
    def __init__(self,
                 root_center: List[float],
                 root_size: float,
                 max_points_per_node: int = 8,
                 min_size: float = 1e-6,
                 max_depth: int = 32):
        self.root = OctreeNode(root_center, root_size)
        self.max_points_per_node = int(max_points_per_node)
        self.min_size = float(min_size)
        self.max_depth = int(max_depth)
        self.dim = 3

    def _in_bounds(self, coords: List[float], node: OctreeNode) -> bool:
        half = node.size / 2.0
        eps = 1e-12
        return (node.center[0] - half - eps <= coords[0] <= node.center[0] + half + eps and
                node.center[1] - half - eps <= coords[1] <= node.center[1] + half + eps and
                node.center[2] - half - eps <= coords[2] <= node.center[2] + half + eps)

    def _octant(self, coords: List[float], center: List[float]) -> int:
        o = 0
        if coords[0] > center[0]: o |= 1 << 0
        if coords[1] > center[1]: o |= 1 << 1
        if coords[2] > center[2]: o |= 1 << 2
        return o

    def insert(self, point: Dict, node: Optional[OctreeNode] = None, depth: int = 0) -> None:
        if node is None:
            node = self.root

        coords = point.get('coords')
        if coords is None or len(coords) != self.dim:
            raise ValueError("Point must have 'coords' with length 3.")

        if not self._in_bounds(coords, node):
            return

        if node.is_leaf:
            if (len(node.data) < self.max_points_per_node) or (node.size <= self.min_size) or (depth >= self.max_depth):
                node.data.append(point)
                return

            all_pts = node.data + [point]
            first_oct = self._octant(all_pts[0]['coords'], node.center)
            if all(self._octant(p['coords'], node.center) == first_oct for p in all_pts):
                node.data.append(point)
                return

            node.is_leaf = False
            old_points = node.data
            node.data = []

            for p in old_points:
                self._insert_child(p, node, depth)
            self._insert_child(point, node, depth)
        else:
            self._insert_child(point, node, depth)

    def _insert_child(self, point: Dict, node: OctreeNode, depth: int) -> None:
        octant = self._octant(point['coords'], node.center)
        if node.children[octant] is None:
            half = node.size / 2.0
            new_center = [
                node.center[i] + (((octant >> i) & 1) * half - half / 2.0)
                for i in range(3)
            ]
            node.children[octant] = OctreeNode(new_center, half)

        self.insert(point, node.children[octant], depth + 1)

    def kNN(self, target: List[float], k: int) -> List[dict]:
        heap: List[Tuple[float, int, dict]] = []

        def sqdist(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

        def bbox_min_sqdist(pt, center, size):
            half, d2 = size / 2.0, 0.0
            for i in range(3):
                lo, hi = center[i] - half, center[i] + half
                if pt[i] < lo:
                    d2 += (lo - pt[i]) ** 2
                elif pt[i] > hi:
                    d2 += (pt[i] - hi) ** 2
            return d2

        def search(node: Optional[OctreeNode]) -> None:
            if node is None:
                return

            min_d2 = bbox_min_sqdist(target, node.center, node.size)
            if len(heap) == k and min_d2 > -heap[0][0]:
                return

            if node.is_leaf:
                for p in node.data:
                    d2 = sqdist(p['coords'], target)
                    item = (-d2, id(p), p)
                    if len(heap) < k:
                        heapq.heappush(heap, item)
                    elif d2 < -heap[0][0]:
                        heapq.heappushpop(heap, item)
            else:
                children = []
                for ch in node.children:
                    if ch is not None:
                        children.append((bbox_min_sqdist(target, ch.center, ch.size), ch))
                children.sort(key=lambda x: x[0])

                for _, ch in children:
                    search(ch)

        search(self.root)
        return [t[2] for t in sorted(heap, key=lambda x: -x[0])]

    def _intersects(self, node: OctreeNode, mn: List[float], mx: List[float]) -> bool:
        half = node.size / 2.0
        if node.center[0] + half < mn[0] or node.center[0] - half > mx[0]:
            return False
        if node.center[1] + half < mn[1] or node.center[1] - half > mx[1]:
            return False
        if node.center[2] + half < mn[2] or node.center[2] - half > mx[2]:
            return False
        return True

    def range_query(self, mn: List[float], mx: List[float], node: Optional[OctreeNode] = None, out: Optional[List[dict]] = None) -> List[dict]:
        if out is None:
            out = []
        if node is None:
            node = self.root
        if not self._intersects(node, mn, mx):
            return out

        if node.is_leaf:
            for p in node.data:
                c = p['coords']
                if mn[0] <= c[0] <= mx[0] and mn[1] <= c[1] <= mx[1] and mn[2] <= c[2] <= mx[2]:
                    out.append(p)
        else:
            for child in node.children:
                if child is not None:
                    self.range_query(mn, mx, child, out)
        return out
