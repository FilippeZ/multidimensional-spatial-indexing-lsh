"""
R-Tree implementation for arbitrary k-dimensional spatial partitioning.
Uses Minimum Bounding Rectangles (MBR) for region pruning and proximity queries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import heapq


def _prod(iterable) -> float:
    acc = 1.0
    for v in iterable:
        acc *= v
    return acc


def _bbox_min_dist_sq(target: Tuple[float, ...], mins: List[float], maxs: List[float]) -> float:
    d2 = 0.0
    for t, lo, hi in zip(target, mins, maxs):
        if t < lo:
            d2 += (lo - t) ** 2
        elif t > hi:
            d2 += (t - hi) ** 2
    return d2


@dataclass
class RTreeEntryK:
    mins: List[float]
    maxs: List[float]
    data: Optional[Tuple[float, ...]] = None

    def enlarge(self, other: "RTreeEntryK") -> "RTreeEntryK":
        return RTreeEntryK(
            [min(a, b) for a, b in zip(self.mins, other.mins)],
            [max(a, b) for a, b in zip(self.maxs, other.maxs)],
            self.data
        )

    def volume(self) -> float:
        eps = 1e-12
        sides = [max(eps, b - a) for a, b in zip(self.mins, self.maxs)]
        return float(_prod(sides))

    def intersects(self, qmins: List[float], qmaxs: List[float]) -> bool:
        return all(not (self.maxs[i] < qmins[i] or self.mins[i] > qmaxs[i])
                   for i in range(len(self.mins)))


@dataclass
class RTreeNodeK:
    entries: List[Any] = field(default_factory=list)
    leaf: bool = True
    bounding: Optional[RTreeEntryK] = None

    def update_bounding(self) -> None:
        if not self.entries:
            self.bounding = None
            return

        if self.leaf:
            b0: RTreeEntryK = self.entries[0]
            mins = list(b0.mins); maxs = list(b0.maxs)
            for e in self.entries[1:]:
                mins = [min(m, v) for m, v in zip(mins, e.mins)]
                maxs = [max(m, v) for m, v in zip(maxs, e.maxs)]
        else:
            b0: RTreeEntryK = self.entries[0].bounding
            mins = list(b0.mins); maxs = list(b0.maxs)
            for child in self.entries[1:]:
                bc = child.bounding
                mins = [min(m, v) for m, v in zip(mins, bc.mins)]
                maxs = [max(m, v) for m, v in zip(maxs, bc.maxs)]
        self.bounding = RTreeEntryK(mins, maxs, data=None)

    def enlarge_bounding(self, entry: RTreeEntryK) -> None:
        if self.bounding is None:
            self.bounding = RTreeEntryK(list(entry.mins), list(entry.maxs), data=None)
        else:
            self.bounding = RTreeEntryK(
                [min(a, b) for a, b in zip(self.bounding.mins, entry.mins)],
                [max(a, b) for a, b in zip(self.bounding.maxs, entry.maxs)],
                data=None
            )


class RTreeK:
    def __init__(self, k: int, max_entries: int = 8) -> None:
        if k <= 0 or max_entries < 2:
            raise ValueError("Invalid k or max_entries parameters")
        self.k = k
        self.max_entries = max_entries
        self.root = RTreeNodeK()

    def insert(self, point: Tuple[float, ...]) -> None:
        entry = RTreeEntryK(list(point), list(point), data=point)
        split_node = self._insert_node(self.root, entry)
        if split_node:
            new_root = RTreeNodeK(leaf=False)
            new_root.entries = [self.root, split_node]
            new_root.update_bounding()
            self.root = new_root

    def _insert_node(self, node: RTreeNodeK, entry: RTreeEntryK) -> Optional[RTreeNodeK]:
        node.enlarge_bounding(entry)
        if node.leaf:
            node.entries.append(entry)
            if len(node.entries) > self.max_entries:
                return self._split_leaf(node)
            return None
        else:
            best_child = self._choose_subtree(node, entry)
            split_child = self._insert_node(best_child, entry)
            node.update_bounding()
            if split_child:
                node.entries.append(split_child)
                node.update_bounding()
                if len(node.entries) > self.max_entries:
                    return self._split_internal(node)
            return None

    def _choose_subtree(self, node: RTreeNodeK, entry: RTreeEntryK) -> RTreeNodeK:
        best_child = None
        min_enlargement = float("inf")
        min_vol = float("inf")
        for child in node.entries:
            vol_before = child.bounding.volume()
            enlarged = child.bounding.enlarge(entry)
            vol_after = enlarged.volume()
            enlargement = vol_after - vol_before
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                min_vol = vol_before
                best_child = child
            elif abs(enlargement - min_enlargement) < 1e-12:
                if vol_before < min_vol:
                    min_vol = vol_before
                    best_child = child
        return best_child

    def _split_leaf(self, node: RTreeNodeK) -> RTreeNodeK:
        entries = node.entries
        node.entries = entries[: len(entries) // 2]
        node.update_bounding()
        sibling = RTreeNodeK(leaf=True)
        sibling.entries = entries[len(entries) // 2 :]
        sibling.update_bounding()
        return sibling

    def _split_internal(self, node: RTreeNodeK) -> RTreeNodeK:
        entries = node.entries
        node.entries = entries[: len(entries) // 2]
        node.update_bounding()
        sibling = RTreeNodeK(leaf=False)
        sibling.entries = entries[len(entries) // 2 :]
        sibling.update_bounding()
        return sibling

    def kNN(self, target: Tuple[float, ...], k: int) -> List[Tuple[float, ...]]:
        heap: List[Tuple[float, Tuple[float, ...]]] = []
        pq: List[Tuple[float, Any]] = []

        if self.root.bounding:
            d2 = _bbox_min_dist_sq(target, self.root.bounding.mins, self.root.bounding.maxs)
            heapq.heappush(pq, (d2, self.root))

        while pq:
            d2_node, obj = heapq.heappop(pq)

            if len(heap) == k and d2_node >= -heap[0][0]:
                break

            if isinstance(obj, RTreeNodeK):
                if obj.leaf:
                    for e in obj.entries:
                        dist_pt = sum((target[i] - e.data[i]) ** 2 for i in range(self.k))
                        item = (-dist_pt, e.data)
                        if len(heap) < k:
                            heapq.heappush(heap, item)
                        elif dist_pt < -heap[0][0]:
                            heapq.heapreplace(heap, item)
                else:
                    for child in obj.entries:
                        if child.bounding:
                            cd2 = _bbox_min_dist_sq(target, child.bounding.mins, child.bounding.maxs)
                            if len(heap) < k or cd2 < -heap[0][0]:
                                heapq.heappush(pq, (cd2, child))

        heap_sorted = sorted(heap, key=lambda t: -t[0])
        return [p for _, p in heap_sorted]

    def range_query(self, query_ranges: List[Tuple[float, float]]) -> List[Tuple[float, ...]]:
        qmins = [r[0] for r in query_ranges]
        qmaxs = [r[1] for r in query_ranges]
        out: List[Tuple[float, ...]] = []

        def search(node: RTreeNodeK):
            if node.bounding and not node.bounding.intersects(qmins, qmaxs):
                return

            if node.leaf:
                for e in node.entries:
                    if e.intersects(qmins, qmaxs):
                        out.append(e.data)
            else:
                for child in node.entries:
                    search(child)

        search(self.root)
        return out
