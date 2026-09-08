"""Bounded component association before per-sphere tracking (no prediction)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Shape:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    count: int
    seen: float


class ComponentAssociator:
    def __init__(self, distance=.3, size_ratio=.4, voxel_size=.05, ttl=.3):
        self.distance, self.size_ratio, self.voxel_size, self.ttl = distance, size_ratio, voxel_size, ttl
        self.shapes = {}
        self.next_id = 0

    def clear(self):
        self.shapes.clear()

    def update(self, components, now):
        self.shapes = {key: value for key, value in self.shapes.items() if now - value.seen <= self.ttl}
        current = {}
        for component in components:
            points = component.voxel_centers
            if len(points) and np.isfinite(points).all():
                current[component.component_id] = Shape(points.mean(axis=0), points.min(axis=0),
                    points.max(axis=0), len(points), now)
        pairs = []
        for index, shape in current.items():
            for key, previous in self.shapes.items():
                distance = float(np.linalg.norm(shape.center - previous.center))
                count_ratio = min(shape.count, previous.count) / max(shape.count, previous.count)
                extent = shape.upper - shape.lower + self.voxel_size
                old_extent = previous.upper - previous.lower + self.voxel_size
                shape_ratio = float(np.min(np.minimum(extent, old_extent) / np.maximum(extent, old_extent)))
                if distance > self.distance or min(count_ratio, shape_ratio) < self.size_ratio:
                    continue
                intersection = np.prod(np.maximum(0., np.minimum(shape.upper, previous.upper)
                    - np.maximum(shape.lower, previous.lower) + self.voxel_size))
                union = np.prod(extent) + np.prod(old_extent) - intersection
                iou = float(intersection / max(union, 1e-12))
                score = distance + .1 * (1. - iou) + .05 * (1. - count_ratio)
                pairs.append((score, index, key))
        assigned, used, translations = {}, set(), {}
        for _, index, key in sorted(pairs):
            if index in assigned or key in used:
                continue
            assigned[index] = key
            used.add(key)
            translations[index] = current[index].center - self.shapes[key].center
        for index, shape in current.items():
            if index not in assigned:
                assigned[index] = self.next_id
                self.next_id += 1
                translations[index] = np.zeros(3)
            self.shapes[assigned[index]] = shape
        return assigned, translations
