"""Reading an export (explore_demo.py --export DIR) for the dense-model tools (splat.py, tsdf.py).

    run = RunData(EXPORT)
    run.K, run.width, run.height       the camera (OpenCV: x right, y down, z forward)
    run.true_poses, run.slam_poses     {frame: 4x4 camera-to-world}, both in the true (Replica) world;
                                       slam_poses has only the frames pySLAM tracked
    run.rgb(k), run.depth(k)           a frame's colour (uint8 RGB) and depth (metres, 0 = no depth)
    run.true_points()                  the true geometry: points back-projected from the true depth
                                       (thinned to one per 2 cm), N x 3, and their colours

Only NumPy and OpenCV, so that it runs at pySLAM's default level and at the full level.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_tum(path: Path) -> dict[float, np.ndarray]:
    """A TUM trajectory (t x y z qx qy qz qw) as {t: 4x4 camera-to-world}."""
    out = {}
    for row in np.loadtxt(path, comments="#", ndmin=2):
        t, p, q = row[0], row[1:4], row[4:8]
        x, y, z, w = q / np.linalg.norm(q)
        T = np.eye(4)
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        T[:3, 3] = p
        out[round(float(t), 6)] = T
    return out


class RunData:
    def __init__(self, folder: str | Path):
        self.dir = Path(folder)
        if not (self.dir / "cam_params.json").exists():
            raise SystemExit(f"{self.dir} is not an export (no cam_params.json): see explore_demo.py --export")
        cam = json.loads((self.dir / "cam_params.json").read_text())["camera"]
        self.width, self.height = int(cam["w"]), int(cam["h"])
        self.K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
        self.depth_scale = float(cam["scale"])
        with open(self.dir / "frames.csv", newline="") as f:
            self.frames = [dict(frame=int(r["frame"]), step=int(r["step"]), t=float(r["t"]), state=r["state"])
                           for r in csv.DictReader(f)]
        traj = np.loadtxt(self.dir / "traj.txt").reshape(-1, 4, 4)
        self.true_poses = {k: traj[k] for k in range(len(traj))}
        final = self.dir / "slam" / "trajectory_final.txt"
        tum = read_tum(final) if final.exists() and final.stat().st_size > 0 else {}
        self.slam_poses = {fr["frame"]: tum[round(fr["t"], 6)] for fr in self.frames if round(fr["t"], 6) in tum}

    def poses(self, which: str) -> dict[int, np.ndarray]:
        if which == "true":
            return self.true_poses
        if which == "slam":
            return self.slam_poses
        raise ValueError(f"poses: 'true' or 'slam', not {which!r}")

    def rgb(self, k: int) -> np.ndarray:
        bgr = cv2.imread(str(self.dir / "results" / f"frame{k:06d}.jpg"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, k: int) -> np.ndarray:
        d = cv2.imread(str(self.dir / "results" / f"depth{k:06d}.png"), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / self.depth_scale

    def true_points(self) -> tuple[np.ndarray, np.ndarray]:
        """The export's true point cloud (colmap_gt's points: the true depth, one point per 2 cm)."""
        path = self.dir / "colmap_gt" / "sparse" / "0" / "points3D.txt"
        rows = np.loadtxt(path, comments="#", usecols=(1, 2, 3, 4, 5, 6), ndmin=2)
        return rows[:, :3], rows[:, 3:6].astype(np.uint8)


def backproject(depth: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, stride: int = 1,
                max_depth: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """The world points of a depth image's pixels (every stride-th), and their (row, column)."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    ok = (z > 0) & (z < max_depth)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1) @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    return P, np.stack([v, u], 1)
