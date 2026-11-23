"""Closed loop: a simulated robot explores a Replica scene while pySLAM tracks it.

    cd pyslam && pixi run python ../sim/explore_demo.py --frames 600 [--gui]

Each step: the robot's true pose -> rendered RGB-D -> slam.track() -> the controller
reads the measurement and SLAM's estimate and returns a command -> the robot moves.
Simulated time advances by --dt per frame however long the frame took to compute.

--gui opens three windows while it runs: pySLAM's 3D Map Viewer, the rendered camera image
with the tracked features, and a top-down view of the robot's true path (green) and pySLAM's
estimate (red). At the end they stay open until the Map Viewer is closed or q / Esc is pressed
in an image window (which also stops a run early); --no-hold closes them at once.
"""

from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyslam"))
sys.path.insert(0, str(ROOT / "replica" / "wgpu-renderer"))
sys.path.insert(0, str(ROOT / "sim"))

import pyslam.config  # noqa: E402,F401  (first: sets the native library paths)
import numpy as np  # noqa: E402
from pyslam.config import Config  # noqa: E402
from pyslam.config_parameters import Parameters  # noqa: E402
from pyslam.io.dataset_types import DatasetEnvironmentType, SensorType  # noqa: E402
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs  # noqa: E402
try:  # pySLAM from its #144 on (main): pyslam.slam exports Slam and camera_from_config
    from pyslam.slam import Slam, camera_from_config  # noqa: E402
except ImportError:  # older pySLAM, e.g. the course's comp0227-2026-lab05: the C++ core's pinhole camera
    from pyslam.slam import PinholeCamera as camera_from_config  # noqa: E402
    from pyslam.slam.slam import Slam  # noqa: E402
from replica_wgpu import Camera, Renderer  # noqa: E402
from replica_wgpu.scene import read_ply  # noqa: E402

import export  # noqa: E402
from lockstep_demo import DEPTH_SCALE, LOCKSTEP_PARAMETERS  # noqa: E402
from floor_map import FloorMap  # noqa: E402
from robot import Estimate, Observation, Robot, Wander  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(ROOT / "data" / "replica" / "room_0"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "replica_room0.yaml"))
    ap.add_argument("--start", default=str(ROOT / "configs" / "start_room0.txt"),
                    help="file of camera-to-world poses; the robot starts at the first, levelled")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--dt", type=float, default=1 / 25, help="simulated seconds per frame")
    ap.add_argument("--height", type=float, default=1.2, help="camera height above the floor, m")
    ap.add_argument("--stop-at", type=float, default=1.5,
                    help="the example controller turns away when something is closer than this, m")
    ap.add_argument("--start-margin", type=float, default=0.0,
                    help="start at least this far (m) inside the free floor (default 0: the nearest free cell, which "
                         "fixed routes are planned from; 0.3 keeps Wander from being trapped on the edge of the free floor "
                         "by low furniture its camera cannot see, e.g. from pose 400 of NICE-SLAM's room0 path)")
    ap.add_argument("--no-floor-map", action="store_true",
                    help="the old robot: no collisions with what lies below the camera's view (it can drive over "
                         "low furniture), and the start exactly at --start's first pose (above the furniture)")
    ap.add_argument("--controller", choices=("wander", "script"), default="wander",
                    help="wander (the example) or script (sim/scripted.py: play the moves of --script)")
    ap.add_argument("--script", help="script: the file of moves to play (see sim/scripted.py)")
    ap.add_argument("--controller-poses", choices=("slam", "true"), default="slam",
                    help="the poses the controller is given: pySLAM's (default) or the true ones (the ideal case)")
    ap.add_argument("--features", type=int, default=1000)
    ap.add_argument("--adapter", help="GPU for the renderer")
    ap.add_argument("--atlas", choices=["rgba16float", "rgb9e5"], default="rgba16float")
    ap.add_argument("--sheet", help="save a contact sheet of every 50th view to this file")
    ap.add_argument("--gui", action="store_true",
                    help="show the Map Viewer, the camera image with its features and a top-down view")
    ap.add_argument("--no-hold", action="store_true", help="with --gui: close the windows at the end")
    ap.add_argument("--loop-closing", action="store_true",
                    help="run pySLAM's loop closing (off by default: it runs in threads and a process of its own, "
                         "so runs are no longer repeatable)")
    export.add_arguments(ap)
    args = ap.parse_args()

    for name, value in LOCKSTEP_PARAMETERS.items():
        setattr(Parameters, name, value)
    loop_config = None
    if args.loop_closing:  # main_slam.py's default: DBoW3 on the ORB features
        from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
        Parameters.kUseLoopClosing = True
        loop_config = LoopDetectorConfigs.DBOW3
    config = Config(args.config)
    slam = Slam(camera_from_config(config), dict(FeatureTrackerConfigs.ORB2, num_features=args.features),
                loop_config, None, SensorType.RGBD, environment_type=DatasetEnvironmentType.INDOOR,
                config=config, headless=True)
    renderer = Renderer(args.scene, Camera.replica_pyslam(), adapter=args.adapter, atlas_format=args.atlas)
    gui = Gui() if args.gui else None
    exporter = export.from_arguments(args, renderer)

    robot = Robot.from_camera_pose(np.loadtxt(args.start).reshape(-1, 4, 4)[0])
    blocked = None
    if args.no_floor_map:
        floor = float(np.percentile(read_ply(str(Path(args.scene) / "mesh.ply"))[0][:, 2], 1))
        robot.camera_height = floor + args.height
    else:  # the simulator's physics: the robot collides with the floor map, and starts on free floor
        floor = float(np.percentile(read_ply(str(Path(args.scene) / "mesh.ply"))[0][:, 2], 1))
        robot.camera_height = floor + args.height
        floor_map = FloorMap.from_mesh(args.scene, robot.camera_height, robot.clearance)
        x, y = floor_map.nearest_free(robot.x, robot.y, margin=args.start_margin)
        print(f"floor map: {floor_map.area:.1f} m2 of free floor; start moved {np.hypot(x - robot.x, y - robot.y):.2f} m "
              f"onto it", flush=True)
        robot.x, robot.y = x, y
        blocked = floor_map.blocked
    controller = Wander(stop_at=args.stop_at)
    if args.controller == "script":
        from scripted import Scripted
        if not args.script:
            ap.error("--controller script needs --script FILE")
        controller = Scripted(args.script)
    truth, estimates, states, views, wall = [], [], [], [], []
    bumped, bumps = False, 0
    try:
        for i in range(args.frames):
            t0 = time.perf_counter()
            T_true = robot.camera_pose()
            rgb, depth = renderer.render(T_true)
            observation = Observation(i * args.dt, rgb, depth, bumped)

            depth16 = np.clip(depth * DEPTH_SCALE + 0.5, 0, 65535).astype(np.uint16)
            slam.track(np.ascontiguousarray(rgb[:, :, ::-1]), None, depth16, i, observation.time)
            tracking = slam.tracking
            T_est = None
            if tracking.cur_R is not None and tracking.cur_t is not None:
                T_est = np.eye(4)
                T_est[:3, :3], T_est[:3, 3] = tracking.cur_R, np.asarray(tracking.cur_t).reshape(3)
            estimate = Estimate(tracking.state.name, T_est, slam.map.num_keyframes(), slam.map.num_points())

            if args.controller_poses == "true":  # the ideal case: the controller is given the true pose
                command = controller.step(observation, Estimate("OK", T_true, estimate.num_keyframes, estimate.num_points))
            else:
                command = controller.step(observation, estimate)
            bumped = robot.step(command, args.dt, depth, blocked)
            bumps += bumped

            truth.append(T_true)
            estimates.append(T_est if T_est is not None else np.full((4, 4), np.nan))
            states.append(estimate.state)
            wall.append(time.perf_counter() - t0)
            if exporter:  # after the step's timing: writing files is not part of the loop
                export.export_step(exporter, renderer, i, rgb, depth, T_true, estimate.state, T_est)
            if args.sheet and i % 50 == 0:
                views.append(rgb[::4, ::4].copy())
            if (i + 1) % 100 == 0:
                print(f"frame {i + 1}/{args.frames}: {estimate.state}, {estimate.num_points} points, "
                      f"{estimate.num_keyframes} keyframes, travelled {robot.distance:.1f} m", flush=True)
            if gui and not gui.show(slam, rgb, truth, estimates, states, i, args.frames, robot.distance):
                break  # closed by the user
        if hasattr(controller, "summary"):
            print(controller.summary(), flush=True)
            if args.export:
                Path(args.export).mkdir(parents=True, exist_ok=True)
                (Path(args.export) / "controller.json").write_text(json.dumps(controller.events, indent=1))
        if exporter:
            print(f"writing the export to {args.export} ...", flush=True)
            exporter.finish(slam, {"script": "explore_demo.py", **vars(args)})
        if gui and len(states) == args.frames and not args.no_hold:
            print("the windows stay open: close the Map Viewer, or press q or Esc in an image window", flush=True)
            gui.hold(slam)
    finally:
        if gui:
            gui.close()
        slam.quit()

    truth, est = np.stack(truth), np.stack(estimates)
    ref = np.linalg.inv(truth[0]) @ truth
    ok = np.array([s == "OK" for s in states]) & ~np.isnan(est[:, 0, 3])
    not_ok = [i for i, s in enumerate(states) if s not in ("OK", "NOT_INITIALIZED")]
    err = np.linalg.norm(est[ok, :3, 3] - ref[ok, :3, 3], axis=1)
    ms = 1000 * np.array(wall)
    xy = truth[:, :2, 3]
    frames = len(states)
    print(f"\nframes {frames} ({frames * args.dt:.0f} simulated seconds), state OK in {states.count('OK')}, "
          f"other states: {sorted(set(states) - {'OK'})}")
    print(f"travelled {robot.distance:.1f} m, bumper stops {bumps}, area covered "
          f"{np.ptp(xy[:, 0]):.1f} x {np.ptp(xy[:, 1]):.1f} m")
    if not_ok:
        print(f"tracking first lost at frame {not_ok[0]} ({not_ok[0] * args.dt:.1f} s); "
              f"OK again later: {any(s == 'OK' for s in states[not_ok[0]:])}")
    print(f"position error while tracking (no alignment): rmse {1000 * np.sqrt((err ** 2).mean()):.1f} mm, "
          f"max {1000 * err.max():.1f} mm")
    print(f"step median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms: "
          f"{frames * args.dt / (ms.sum() / 1000):.2f}x real time")
    if args.sheet and views:
        from PIL import Image
        cols = 6
        h, w = views[0].shape[:2]
        sheet = np.zeros((h * -(-len(views) // cols), w * cols, 3), np.uint8)
        for k, v in enumerate(views):
            sheet[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = v
        Image.fromarray(sheet).save(args.sheet)


class Gui:
    """The windows of --gui. Their imports are here so that a run without --gui does not load them."""

    TOP_NAME = "Top-down: true path and estimate"
    TOP_SIZE, TOP_MIN_RANGE = 700, 3.0  # top-down window: first size in pixels; it shows at least 3 m across

    def __init__(self):
        import cv2
        from pyslam.viz.viewer3D import Viewer3D

        self.cv2 = cv2
        self.viewer3D = Viewer3D(scale=0.1)
        self.viewer3D.wait_for_ready()
        # resizable, and drawn at the window's own size each frame, so that a bigger window is sharper
        # (the zoom of OpenCV's own window only magnifies the pixels)
        cv2.namedWindow(self.TOP_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.TOP_NAME, self.TOP_SIZE, self.TOP_SIZE)

    def show(self, slam, rgb, truth, estimates, states, i, frames, distance) -> bool:
        """Draw frame i; False when the user closed the windows."""
        cv2 = self.cv2
        self.viewer3D.draw_slam_map(slam)
        img = slam.map.draw_feature_trails(np.ascontiguousarray(rgb[:, :, ::-1]), with_level_radius=False,
                                           trail_max_length=Parameters.kMaxFeatureTrailLength)
        cv2.putText(img, f"frame {i + 1}/{frames}  {states[-1]}  {slam.map.num_keyframes()} keyframes  "
                    f"travelled {distance:.1f} m", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Camera (rendered, with tracked features)", img)
        cv2.imshow(self.TOP_NAME, self.top_down(truth, estimates, states))
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27) and not self.viewer3D.is_closed()

    def top_down(self, truth, estimates, states):
        """The true path (green) and pySLAM's estimate while tracking (red), from above, in the first camera's
        frame (x right, z forward), on a 1 m grid, fitted to the paths and drawn at the window's size."""
        cv2 = self.cv2
        try:
            _, _, w, h = cv2.getWindowImageRect(self.TOP_NAME)
        except cv2.error:
            w = h = 0
        w, h = (w, h) if w >= 100 and h >= 100 else (self.TOP_SIZE, self.TOP_SIZE)
        img = np.full((h, w, 3), 255, np.uint8)

        T0_inv = np.linalg.inv(truth[0])
        true_xz = np.array([(T0_inv @ T)[[0, 2], 3] for T in truth])
        est_xz = np.array([T[[0, 2], 3] if st == "OK" else (np.nan, np.nan) for T, st in zip(estimates, states)])
        both = np.vstack([true_xz, est_xz])
        both = both[np.isfinite(both).all(axis=1)]
        lo, hi = both.min(axis=0) - 0.5, both.max(axis=0) + 0.5  # half a metre round both paths
        centre = (lo + hi) / 2
        span = max(self.TOP_MIN_RANGE, (hi - lo)[0] * h / w, (hi - lo)[1])  # metres up the window
        s = h / span  # pixels per metre
        SHIFT = 4  # sub-pixel drawing: coordinates in 1/16 pixel

        def px(x, z):  # x right, z up the window
            return (np.stack([w / 2 + s * (np.asarray(x) - centre[0]), h / 2 - s * (np.asarray(z) - centre[1])],
                             axis=-1) * (1 << SHIFT)).round().astype(np.int32)

        x0, x1 = centre[0] - w / s / 2, centre[0] + w / s / 2
        z0, z1 = centre[1] - span / 2, centre[1] + span / 2
        for m in range(int(np.floor(x0)), int(np.ceil(x1)) + 1):
            cv2.line(img, tuple(px(m, z0)), tuple(px(m, z1)), (225, 225, 225), 1, cv2.LINE_AA, SHIFT)
        for m in range(int(np.floor(z0)), int(np.ceil(z1)) + 1):
            cv2.line(img, tuple(px(x0, m)), tuple(px(x1, m)), (225, 225, 225), 1, cv2.LINE_AA, SHIFT)
        thick = max(2, round(h / 350))
        # the true path wider, under the estimate, so that it shows where the two agree
        for path, colour, width in ((true_xz, (0, 160, 0), thick + 3), (est_xz, (0, 0, 220), thick)):
            ok = np.isfinite(path).all(axis=1)
            # one polyline per stretch of tracking, so that a gap is not drawn as a jump
            for run in np.split(np.arange(len(path)), np.flatnonzero(np.diff(ok.astype(int))) + 1):
                if ok[run[0]] and len(run) > 1:
                    cv2.polylines(img, [px(path[run, 0], path[run, 1])], False, colour, width, cv2.LINE_AA, SHIFT)
        font = 0.5 * max(1.0, h / 700)
        for row, (text, colour) in enumerate((("true path", (0, 160, 0)), ("pySLAM estimate", (0, 0, 220)),
                                              ("grid: 1 m", (120, 120, 120)))):
            cv2.putText(img, text, (10, int((20 + 20 * row) * font / 0.5)), cv2.FONT_HERSHEY_SIMPLEX, font, colour,
                        1, cv2.LINE_AA)
        return img

    def hold(self, slam):
        while not self.viewer3D.is_closed():
            self.viewer3D.draw_slam_map(slam)
            if (self.cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                break

    def close(self):
        self.cv2.destroyAllWindows()
        self.viewer3D.quit()


if __name__ == "__main__":
    main()
