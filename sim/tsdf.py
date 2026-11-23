"""A mesh of a simulator run by TSDF fusion, with the true poses or with pySLAM's (Lab 06).

    pixi run python ../sim/tsdf.py EXPORT --poses true
    pixi run python ../sim/tsdf.py EXPORT --poses slam

EXPORT is a folder written by `explore_demo.py --export EXPORT`. Each frame's depth image is fused into a
truncated signed distance field (TSDF) on a grid of small cells (voxels), placed with the frame's pose;
the surface is where the distance changes sign, and is extracted as a coloured triangle mesh. Unlike a
splat, this needs no training and no GPU: a few seconds to a minute on a CPU.

By default only the frames that pySLAM tracked are fused, with either set of poses, so that the two meshes
differ only in their poses. The mesh is scored against the true geometry of the same frames (their true
depth placed with their true poses):
    accuracy       how far the mesh is from the true surface (median distance of its vertices), and
                   the share of its vertices within 5 cm
    completeness   the share of the true surface within 5 cm of the mesh

Writes, in OUT (default EXPORT/tsdf_<poses>): mesh.ply (any mesh viewer opens it) and metrics.json.
A window shows the mesh: press E to colour it by its error (white within 1 cm, red at 10 cm or more), C
for its own colours, Q or Esc to close.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_data import RunData, backproject  # noqa: E402

GREEN, RESET = "\033[92m", "\033[0m"


def error_colours(d: np.ndarray, good: float = 0.01, bad: float = 0.10) -> np.ndarray:
    x = np.clip((d - good) / (bad - good), 0, 1)[:, None]
    return (1 - x) * np.array([1.0, 1.0, 1.0]) + x * np.array([0.85, 0.05, 0.05])


def align_rigid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The rotation and translation (4x4) which move the points A closest to B (least squares)."""
    a, b = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - b).T @ (A - a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    T = np.eye(4)
    T[:3, :3] = U @ D @ Vt
    T[:3, 3] = b - T[:3, :3] @ a
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="a folder written by explore_demo.py --export")
    ap.add_argument("--poses", choices=("true", "slam"), default="true",
                    help="place the depth images with the true poses or with pySLAM's")
    ap.add_argument("--all-frames", action="store_true",
                    help="with --poses true: fuse every saved frame, also those pySLAM did not track")
    ap.add_argument("--voxel", type=float, default=0.02, help="cell size, metres")
    ap.add_argument("--trunc", type=float, default=4.0, help="truncation distance, in cells")
    ap.add_argument("--max-depth", type=float, default=5.0, help="ignore depth beyond this, metres")
    ap.add_argument("--min-weight", type=float, default=2.0,
                    help="keep the surface only where at least this many frames saw it")
    ap.add_argument("--align", action="store_true",
                    help="with --poses slam: first move pySLAM's path as a whole (one rotation and translation)"
                         " onto the true path, as evo's -a does, so that only its errors along the way remain")
    ap.add_argument("--out", help="output folder (default EXPORT/tsdf_<poses>[_aligned])")
    ap.add_argument("--headless", action="store_true", help="no window")
    args = ap.parse_args()

    import open3d as o3d
    import open3d.core as o3c
    from scipy.spatial import cKDTree

    run = RunData(args.export)
    poses = run.poses(args.poses)
    if args.align and args.poses == "slam":
        common = sorted(set(poses) & set(run.true_poses))
        T = align_rigid(np.stack([poses[k][:3, 3] for k in common]),
                        np.stack([run.true_poses[k][:3, 3] for k in common]))
        poses = {k: T @ P for k, P in poses.items()}
        angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"aligned pySLAM's path to the true one: turned by {angle:.1f} degrees, moved by "
              f"{100 * np.linalg.norm(T[:3, 3]):.0f} cm")
    if args.all_frames and args.poses == "true":
        frames = sorted(poses)
    else:
        frames = sorted(set(run.slam_poses) & set(run.true_poses))
    if not frames:
        sys.exit(f"no frames with {args.poses} poses in {args.export}")
    suffix = "_aligned" if args.align and args.poses == "slam" else ""
    out = Path(args.out) if args.out else Path(args.export) / f"tsdf_{args.poses}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    # A sparse grid of blocks of 16^3 voxels, allocated where the depth images put a surface
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=args.voxel, block_resolution=16, block_count=100_000,
        device=o3c.Device("CPU:0"))
    K = o3c.Tensor(run.K, o3c.float64)
    trunc = args.trunc * args.voxel
    t0 = time.time()
    reference = []
    for j, k in enumerate(frames):
        depth = run.depth(k)
        world_to_camera = o3c.Tensor(np.linalg.inv(poses[k]), o3c.float64)
        d = o3d.t.geometry.Image(o3c.Tensor(depth))
        c = o3d.t.geometry.Image(o3c.Tensor(run.rgb(k).astype(np.float32) / 255))
        blocks = grid.compute_unique_block_coordinates(d, K, world_to_camera, 1.0, args.max_depth, trunc / args.voxel)
        grid.integrate(blocks, d, c, K, K, world_to_camera, 1.0, args.max_depth, trunc / args.voxel)
        P, _ = backproject(depth, run.K, run.true_poses[k], stride=4, max_depth=args.max_depth)
        reference.append(P)
        if j % 50 == 0:
            print(f"fused {j + 1}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    mesh = grid.extract_triangle_mesh(weight_threshold=args.min_weight).to_legacy()
    mesh.compute_vertex_normals()
    fuse_time = time.time() - t0
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)

    # Score against the true surface of the same frames, thinned to one point per voxel
    R = np.concatenate(reference)
    _, keep = np.unique(np.floor(R / args.voxel).astype(np.int64), axis=0, return_index=True)
    R = R[keep]
    V = np.asarray(mesh.vertices)
    if len(V) == 0:
        sys.exit("the mesh is empty")
    d_acc, _ = cKDTree(R).query(V, k=1, workers=-1)
    d_comp, _ = cKDTree(V).query(R, k=1, workers=-1)
    metrics = {
        "export": str(Path(args.export).resolve()), "poses": args.poses + suffix, "frames": len(frames),
        "voxel_m": args.voxel, "vertices": len(V), "triangles": len(mesh.triangles),
        "fuse_seconds": round(fuse_time, 1),
        "accuracy_median_m": round(float(np.median(d_acc)), 4),
        "accuracy_within_5cm": round(float((d_acc < 0.05).mean()), 3),
        "completeness_within_5cm": round(float((d_comp < 0.05).mean()), 3),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"{GREEN}Summary: TSDF, {args.poses} poses{' (aligned)' if suffix else ''}, {len(frames)} frames fused in {fuse_time:.0f} s, "
          f"{len(V):,} vertices: accuracy {100 * metrics['accuracy_median_m']:.1f} cm (median), "
          f"{100 * metrics['accuracy_within_5cm']:.0f}% within 5 cm; completeness "
          f"{100 * metrics['completeness_within_5cm']:.0f}% within 5 cm; saved in {out}{RESET}")

    if args.headless:
        return
    own = np.asarray(mesh.vertex_colors).copy()
    err = error_colours(d_acc)

    def show(colours):
        def cb(vis):
            mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
            vis.update_geometry(mesh)
            return False
        return cb

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"TSDF mesh, {args.poses} poses (E: error, C: colour, Q: close)", 1280, 800)
    vis.add_geometry(mesh)
    vis.register_key_callback(ord("E"), show(err))
    vis.register_key_callback(ord("C"), show(own))
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
