"""A ground robot carrying a forward-looking RGB-D camera, and the controller interface.

Nothing here imports pySLAM or the renderer: it is plain NumPy, so it can be tested
and taught on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Command:
    """What a controller asks of the robot for the next step."""
    v: float = 0.0       # forward speed, m/s
    omega: float = 0.0   # turn rate, rad/s (positive turns left, seen from above)


@dataclass
class Observation:
    """What the robot's sensors give at one simulated instant."""
    time: float            # simulated seconds
    rgb: np.ndarray        # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32 metres, 0 where there is none
    bumped: bool = False   # the last forward motion was stopped by an obstacle


@dataclass
class Estimate:
    """What SLAM reports after a frame."""
    state: str                          # pySLAM's SlamState name, e.g. "OK", "LOST"
    T_world_camera: np.ndarray | None   # 4x4 in SLAM's own world (the first camera), or None
    num_keyframes: int = 0
    num_points: int = 0

    @property
    def ok(self) -> bool:
        return self.state == "OK" and self.T_world_camera is not None


class Controller:
    """Decide what to do next. Subclass and override step()."""

    def step(self, observation: Observation, estimate: Estimate) -> Command:
        raise NotImplementedError


@dataclass
class Robot:
    """Unicycle model on the floor plane of a z-up scene; the camera looks along the heading."""
    x: float
    y: float
    yaw: float                 # radians, heading in the world's x-y plane
    camera_height: float       # world z of the camera
    clearance: float = 0.35    # m: forward motion stops when something is closer than this
    max_v: float = 0.5
    max_omega: float = 1.0
    distance: float = field(default=0.0, init=False)

    @classmethod
    def from_camera_pose(cls, T_world_camera: np.ndarray, **kw) -> "Robot":
        """Start where a camera pose is, level, heading along its horizontal viewing direction."""
        fwd = T_world_camera[:3, 2]
        return cls(float(T_world_camera[0, 3]), float(T_world_camera[1, 3]),
                   float(np.arctan2(fwd[1], fwd[0])), float(T_world_camera[2, 3]), **kw)

    def camera_pose(self) -> np.ndarray:
        """Camera-to-world, camera axes x right, y down, z forward."""
        fwd = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = np.cross(down, fwd), down, fwd
        T[:3, 3] = self.x, self.y, self.camera_height
        return T

    def step(self, command: Command, dt: float, depth: np.ndarray | None = None, blocked=None) -> bool:
        """Advance by dt. `depth` (the current view) is the bumper; `blocked(x, y)`, if given, says where the
        robot cannot be (the simulator's floor map: furniture below the camera's view, walls). Returns True
        if either stopped the robot."""
        v = float(np.clip(command.v, -self.max_v, self.max_v))
        omega = float(np.clip(command.omega, -self.max_omega, self.max_omega))
        bumped = False
        if v > 0 and depth is not None and nearest_ahead(depth) < self.clearance + v * dt:
            v, bumped = 0.0, True
        self.yaw += omega * dt
        x, y = self.x + v * dt * np.cos(self.yaw), self.y + v * dt * np.sin(self.yaw)
        if v != 0 and blocked is not None and blocked(x, y):
            v, bumped = 0.0, True
            x, y = self.x, self.y
        self.x, self.y = x, y
        self.distance += abs(v) * dt
        return bumped


def nearest_ahead(depth: np.ndarray, width: float = 0.5, height: float = 0.7) -> float:
    """Smallest valid depth in the central part of the image (a low percentile, to ignore specks)."""
    h, w = depth.shape
    band = depth[int(h * (1 - height) / 2): int(h * (1 + height) / 2),
                 int(w * (1 - width) / 2): int(w * (1 + width) / 2)]
    valid = band[band > 0]
    return float(np.percentile(valid, 2)) if valid.size else 0.0


class Wander(Controller):
    """Example controller: drive forward, turn towards the more open side when blocked,
    and stand still turning slowly whenever SLAM is not tracking (so it can recover)."""

    def __init__(self, v: float = 0.3, omega: float = 0.6, stop_at: float = 0.9, turn_after_bump: int = 30):
        self.v, self.omega, self.stop_at = v, omega, stop_at
        self.turn = 0.0
        self.turn_after_bump = turn_after_bump  # steps of turning after a bump before trying forward again
        self.turning = 0

    def step(self, observation: Observation, estimate: Estimate) -> Command:
        if not estimate.ok and estimate.state != "NOT_INITIALIZED":
            return Command(0.0, 0.3)
        depth = observation.depth
        if observation.bumped:
            # stopped by something the depth ahead may not show (furniture below the camera's view): keep
            # turning one way for a while, else the robot alternates between a step forward and a step of
            # turning, and stays where it is
            self.turning = self.turn_after_bump
        if self.turning > 0:
            self.turning -= 1
        elif nearest_ahead(depth) > self.stop_at:
            self.turn = 0.0
            return Command(self.v, 0.0)
        if self.turn == 0.0:   # choose a side once, and keep turning that way until clear
            w = depth.shape[1]
            left, right = depth[:, : w // 3], depth[:, 2 * w // 3:]
            self.turn = self.omega if np.median(left[left > 0]) >= np.median(right[right > 0]) else -self.omega
        return Command(0.0, self.turn)
