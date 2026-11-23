"""Where the robot can be: a floor map of a Replica scene, from its mesh. The simulator's physics.

A cell (CELL metres square) is free if the mesh has floor there and nothing between just above the floor
and just above the camera lies within the robot's clearance of it. Of the free cells, the largest
connected region is the room's main floor; small pockets (between furniture) are left out.

The map comes from the true mesh, so it is the simulator's, not the robot's: a controller must not use it
(it has its own sensors). explore_demo.py uses it to stop the robot where it would hit something, and to
move the start onto free floor.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CELL = 0.05  # m


class FloorMap:
    def __init__(self, main: np.ndarray, free: np.ndarray, x0: float, y0: float, floor: float):
        self.main, self.free, self.x0, self.y0, self.floor = main, free, x0, y0, floor

    @classmethod
    def from_mesh(cls, scene: str | Path, camera_height: float, clearance: float) -> "FloorMap":
        """camera_height: the camera's world z; clearance: how close (m) the robot may come to anything."""
        from replica_wgpu.scene import read_ply

        P = read_ply(str(Path(scene) / "mesh.ply"))[0].astype(np.float64)
        floor = float(np.percentile(P[:, 2], 1))
        x0, y0 = P[:, 0].min(), P[:, 1].min()
        W, H = int((P[:, 0].max() - x0) / CELL) + 1, int((P[:, 1].max() - y0) / CELL) + 1

        def cells(Q):
            return ((Q[:, 1] - y0) / CELL).astype(int), ((Q[:, 0] - x0) / CELL).astype(int)

        has_floor = np.zeros((H, W), np.uint8)
        has_floor[cells(P[P[:, 2] < floor + 0.05])] = 1
        has_floor = cv2.morphologyEx(has_floor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        obstacle = np.zeros((H, W), np.uint8)
        obstacle[cells(P[(P[:, 2] > floor + 0.05) & (P[:, 2] < camera_height + 0.1)])] = 1
        dist = cv2.distanceTransform((1 - obstacle).astype(np.uint8), cv2.DIST_L2, 5) * CELL
        free = (has_floor > 0) & (dist >= clearance)
        _, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=8)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return cls(labels == int(np.argmax(sizes)), free, x0, y0, floor)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.y0) / CELL), int((x - self.x0) / CELL)

    def is_free(self, x: float, y: float) -> bool:
        """Whether the robot can be at (x, y): on the room's main free floor."""
        r, c = self.cell(x, y)
        return 0 <= r < self.main.shape[0] and 0 <= c < self.main.shape[1] and bool(self.main[r, c])

    def blocked(self, x: float, y: float) -> bool:
        return not self.is_free(x, y)

    def nearest_free(self, x: float, y: float, margin: float = 0.0) -> tuple[float, float]:
        """The centre of the free cell nearest to (x, y) that is at least `margin` (m) inside the free floor.
        A start on the edge of the free floor can trap a robot whose camera cannot see what blocks it (low
        furniture close by drops out of a level camera's view): every heading that looks open is blocked."""
        inner = self.main
        if margin > 0:
            d = cv2.distanceTransform(self.main.astype(np.uint8), cv2.DIST_L2, 5) * CELL
            inner = d >= margin
            if not inner.any():
                inner = self.main
        rows, cols = np.nonzero(inner)
        r0, c0 = (y - self.y0) / CELL, (x - self.x0) / CELL
        k = int(np.argmin((rows + 0.5 - r0) ** 2 + (cols + 0.5 - c0) ** 2))
        return self.x0 + (cols[k] + 0.5) * CELL, self.y0 + (rows[k] + 0.5) * CELL

    @property
    def area(self) -> float:
        return float(self.main.sum() * CELL * CELL)
