"""Lockstep loop: pose -> Replica render -> pySLAM track -> read results -> next pose.

    cd pyslam && pixi run python ../sim/lockstep_demo.py --frames 300

A first prototype. The "controller" just replays a recorded path; the point is the
loop itself: one process, a simulated clock that advances a fixed step per frame
whatever the machine's speed, and local mapping inside track() so that the result
does not depend on how fast frames arrive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyslam"))
sys.path.insert(0, str(ROOT / "replica" / "wgpu-renderer"))

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

sys.path.insert(0, str(ROOT / "sim"))
import export  # noqa: E402

DEPTH_SCALE = 6553.5  # settings/REPLICA.yaml DepthMapFactor

# Everything asynchronous is off: with these, a frame is finished when track() returns.
LOCKSTEP_PARAMETERS = {
    "kLocalMappingOnSeparateThread": False,
    "kUseLoopClosing": False,
    "kDoVolumetricIntegration": False,
    "kDoSparseSemanticMappingAndSegmentation": False,
    "kUseLargeWindowBA": False,
    "kUseParallelProcessLBA": False,
    "kLocalMappingParallelKpsMatching": False,
    "kUseMotionBlurDection": False,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(ROOT / "data" / "replica" / "room_0"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "replica_room0.yaml"))
    ap.add_argument("--path", default=str(ROOT / "data" / "nice_slam_ref" / "Replica" / "room0" / "traj.txt"),
                    help="camera-to-world poses to replay, 16 numbers per line")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--dt", type=float, default=1 / 25, help="simulated seconds per frame")
    ap.add_argument("--features", type=int, default=1000)
    ap.add_argument("--adapter", help="GPU for the renderer (see the renderer's README)")
    ap.add_argument("--instant-mapping", action="store_true",
                    help="do not charge local mapping its measured wall time on the simulated clock "
                         "(Parameters.kLocalMappingSimulateBusyTimeInSingleThread = False): repeatable, "
                         "but far more keyframes")
    ap.add_argument("--dump", help="save per-frame poses, states and map sizes to this .npz, "
                                   "to find where two runs first differ")
    ap.add_argument("--atlas", choices=["rgba16float", "rgb9e5"], default="rgba16float")
    export.add_arguments(ap)
    args = ap.parse_args()

    for name, value in LOCKSTEP_PARAMETERS.items():
        setattr(Parameters, name, value)
    if args.instant_mapping:
        Parameters.kLocalMappingSimulateBusyTimeInSingleThread = False

    config = Config(args.config)
    camera = camera_from_config(config)
    tracker = dict(FeatureTrackerConfigs.ORB2, num_features=args.features)
    slam = Slam(camera, tracker, None, None, SensorType.RGBD,
                environment_type=DatasetEnvironmentType.INDOOR, config=config, headless=True)

    renderer = Renderer(args.scene, Camera.replica_pyslam(), adapter=args.adapter, atlas_format=args.atlas)
    truth = np.loadtxt(args.path).reshape(-1, 4, 4)[: args.frames]
    exporter = export.from_arguments(args, renderer)

    t_render, t_track, estimates, states, sizes = [], [], [], [], []
    try:
        for i, T_world_camera in enumerate(truth):       # the simulator's true pose
            sim_time = i * args.dt                         # simulated, not wall, time
            t0 = time.perf_counter()
            rgb, depth = renderer.render(T_world_camera)   # measurements
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            depth16 = np.clip(depth * DEPTH_SCALE + 0.5, 0, 65535).astype(np.uint16)
            t1 = time.perf_counter()
            slam.track(bgr, None, depth16, i, sim_time)    # pySLAM
            t2 = time.perf_counter()
            tracking = slam.tracking                       # results
            states.append(tracking.state.name)
            T = np.full((4, 4), np.nan)
            if tracking.cur_R is not None and tracking.cur_t is not None:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = tracking.cur_R, np.asarray(tracking.cur_t).reshape(3)
            estimates.append(T)
            sizes.append((slam.map.num_keyframes(), slam.map.num_points()))
            t_render.append(t1 - t0)
            t_track.append(t2 - t1)
            if exporter:  # after the timing: writing files is not part of the loop
                export.export_step(exporter, renderer, i, rgb, depth, T_world_camera, states[-1], T)
            # A controller would choose the next pose here, from `T`, the state and slam.map.
            if (i + 1) % 100 == 0:
                print(f"frame {i + 1}/{len(truth)}: {states[-1]}, {slam.map.num_points()} points, "
                      f"{slam.map.num_keyframes()} keyframes", flush=True)
        keyframes, points = slam.map.num_keyframes(), slam.map.num_points()
        if exporter:
            print(f"writing the export to {args.export} ...", flush=True)
            exporter.finish(slam, {"script": "lockstep_demo.py", **vars(args)})
    finally:
        slam.quit()

    # pySLAM's world is the first camera: express the true path in it and compare positions.
    est = np.stack(estimates)
    ref = np.linalg.inv(truth[0]) @ truth
    ok = ~np.isnan(est[:, 0, 3])
    err = np.linalg.norm(est[ok, :3, 3] - ref[ok, :3, 3], axis=1)
    ms_r, ms_t = 1000 * np.array(t_render), 1000 * np.array(t_track)
    print(f"\nframes {len(truth)}, with a pose {ok.sum()}, state OK in {states.count('OK')}; "
          f"{keyframes} keyframes, {points} points")
    print(f"position error against the true path (no alignment): rmse {1000 * np.sqrt((err ** 2).mean()):.1f} mm, "
          f"max {1000 * err.max():.1f} mm")
    if args.dump:
        np.savez(args.dump, poses=est, states=np.array(states), sizes=np.array(sizes))
    import hashlib
    print(f"trajectory digest {hashlib.sha1(np.round(est[ok], 9).tobytes()).hexdigest()[:12]}")
    print(f"render   median {np.median(ms_r):.1f} ms, p95 {np.percentile(ms_r, 95):.1f} ms")
    print(f"track    median {np.median(ms_t):.1f} ms, p95 {np.percentile(ms_t, 95):.1f} ms, max {ms_t.max():.1f} ms")
    print(f"loop     {1000 / (np.median(ms_r) + np.median(ms_t)):.1f} frames per second (median), "
          f"{args.dt * len(truth) / (ms_r.sum() + ms_t.sum()) * 1000:.2f}x real time at dt = {args.dt:.3f} s")


if __name__ == "__main__":
    main()
