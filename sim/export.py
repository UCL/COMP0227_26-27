"""Write a simulator run in the formats other systems read: the NICE-SLAM Replica layout, pySLAM's
results, COLMAP models and nerfstudio transforms.

    exporter = Exporter(out_dir, dt)                    # before the loop
    exporter.add(i, rgb, depth, T_world_camera, state)  # every step, as it is rendered
    exporter.add_estimate(i, T_slam_camera)             # pySLAM's pose of the step as tracked, or None
    exporter.finish(slam, run_info)                     # after the loop, before slam.quit()

With every=n only every n-th step is saved as a frame, to keep the folder small; frames are numbered
0, 1, 2, ... without gaps (the loaders below expect that), and frames.csv gives each one's step. The
trajectories in TUM format keep every step. With objects (Renderer.objects()), add() also takes the
per-pixel object ids from Renderer.render_objects(); exporter.wants(i) says whether step i is saved, so
that the id pass runs only for those.

Layout of out_dir (camera-to-world matrices are 4x4, OpenCV camera axes: x right, y down, z forward):

    results/frame%06d.jpg, results/depth%06d.png   NICE-SLAM's Replica layout: RGB, and depth as uint16
    traj.txt                                         (metres x 6553.5); the true camera-to-world pose of
    cam_params.json                                  each frame, 16 numbers per line; the camera. SplaTAM,
                                                     MonoGS and GS-ICP SLAM read this layout, and so does
                                                     pySLAM (REPLICA_DATASET).
    results/object%06d.png                           with objects: the annotated object at each pixel, as
                                                     uint16 id + 1 (0 = nothing, 1 = a surface with no
                                                     annotation); ids are those of objects.json
    objects.json                                     with objects: the scene's annotated objects, each with
                                                     its class, oriented box in the true world (centre,
                                                     sizes, rotation whose columns are the box's axes, its 8
                                                     corners), volume, and how many saved frames and pixels
                                                     showed it
    frames.csv                                       per frame: frame, step, t, pySLAM's tracking state
    images/frame%06d.jpg                             the same RGB files (hard links when possible), for
                                                     VGGT-SLAM / AMB3R (--image_folder)
    groundtruth.txt                                  true poses in TUM format (t x y z qx qy qz qw), for evo
    slam/trajectory_online.txt                       pySLAM's pose of every tracked step as tracked, and
    slam/trajectory_final.txt                        after the map's later corrections; TUM format
    slam/keyframes.txt                               pySLAM's keyframes: step, t, and pose (TUM order)
    slam/map_points.ply                              pySLAM's map points with their colours
    slam/T_world_slam.txt                            pySLAM's world (its first camera) in the true world
    colmap_gt/sparse/0/{cameras,images,points3D}.txt  COLMAP text models for Gaussian splatting (gsplat,
    colmap_gt/images/                                  3DGS), each with its images/ (hard links): true poses
    colmap_slam/...                                    with points back-projected from the true depth;
                                                       pySLAM's final poses with its map points
    transforms_gt.json, transforms_slam.json         nerfstudio / NeRF (OpenGL camera axes), with depth
    slam_state/                                      with save_map: pySLAM's map (Slam.save_system_state),
                                                     for Slam.load_system_state. Large: pySLAM stores each
                                                     keyframe's images in it as JSON, about 8 MB a keyframe
    run.json                                         what was run (arguments, the three repositories'
                                                     versions) and a summary

Every pose written here is in the true (Replica) world, also pySLAM's: pySLAM's world is its first
camera, so its poses are mapped with T_world_slam = the first true pose. RGB-D tracking is metric, so no
scale is involved, and the two COLMAP models and transforms files are directly comparable.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

DEPTH_SCALE = 6553.5  # NICE-SLAM's Replica depth PNGs and pySLAM's settings/REPLICA.yaml
OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def tum_line(t: float, T: np.ndarray) -> str:
    w, x, y, z = quat_wxyz(T[:3, :3])
    return f"{t:.6f} {T[0, 3]:.9f} {T[1, 3]:.9f} {T[2, 3]:.9f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}"


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float):
    """One point per cubic cell of side voxel (the first in it); voxel <= 0 keeps them all."""
    if voxel <= 0 or len(xyz) == 0:
        return xyz, rgb
    _, keep = np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0, return_index=True)
    keep.sort()
    return xyz[keep], rgb[keep]


def git_version(path: Path) -> str | None:
    """The checkout's commit (git describe --always --dirty), or None outside git."""
    try:
        return subprocess.run(["git", "-C", str(path), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def box_corners(center, sizes, rotation) -> np.ndarray:
    """The 8 corners (8x3) of an oriented box: center + rotation @ (signs * sizes / 2)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    return np.asarray(center, np.float64) + (signs * np.asarray(sizes) / 2) @ np.asarray(rotation).T


def link_or_copy(src: Path, dst: Path) -> None:
    """A hard link (no extra disk space), or a copy where links are not possible."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def write_colmap_bin(d: Path, cam, poses: dict, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """cameras.bin, images.bin, points3D.bin (COLMAP's binary format): one PINHOLE camera (model id 1),
    images without 2D points, points without tracks. Same conventions as the text files."""
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, cam.width, cam.height))
        f.write(struct.pack("<4d", cam.fx, cam.fy, cam.cx + 0.5, cam.cy + 0.5))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(poses)))
        for k, i in enumerate(sorted(poses), start=1):
            Tcw = np.linalg.inv(poses[i])
            f.write(struct.pack("<i4d3di", k, *quat_wxyz(Tcw[:3, :3]), *Tcw[:3, 3], 1))
            f.write(f"frame{i:06d}.jpg".encode() + b"\0")
            f.write(struct.pack("<Q", 0))
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for k, (p, col) in enumerate(zip(xyz, rgb), start=1):
            f.write(struct.pack("<Q3d3BdQ", k, *map(float, p), *map(int, col), 0.0, 0))


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary PLY with float x y z, zero normals (3DGS reads them) and uchar red green blue."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
    vertex = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                                       ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertex)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def add_arguments(ap) -> None:
    """The export options of the demos."""
    ap.add_argument("--export", metavar="DIR", help="write the run for other systems (NICE-SLAM layout, "
                                                    "pySLAM's results, COLMAP, nerfstudio; see sim/export.py)")
    ap.add_argument("--every", type=int, default=1, help="with --export: save every n-th step as a frame")
    ap.add_argument("--objects", action="store_true",
                    help="with --export: also the object id of every pixel, and the scene's objects with their "
                         "boxes (objects.json); costs a second render pass per saved frame")
    ap.add_argument("--save-map", action="store_true",
                    help="with --export: also pySLAM's map (slam_state/; about 8 MB per keyframe)")


def from_arguments(args, renderer) -> "Exporter | None":
    """The Exporter the options ask for, or None."""
    if not args.export:
        return None
    objects = None
    if args.objects:
        if not hasattr(renderer, "render_objects"):
            raise SystemExit("--objects needs a renderer with object ids (Replica-Dataset PR #14)")
        objects = renderer.objects()
    return Exporter(args.export, args.dt, camera=renderer.camera, every=args.every, objects=objects,
                    save_map=args.save_map)


def export_step(exporter, renderer, i, rgb, depth, T_world_camera, state, T_slam_camera) -> None:
    """Hand step i to the exporter, rendering its object ids if they are wanted."""
    ids = renderer.render_objects(T_world_camera) if exporter.objects is not None and exporter.wants(i) else None
    exporter.add(i, rgb, depth, T_world_camera, state, ids)
    exporter.add_estimate(i, T_slam_camera)


class Exporter:
    def __init__(self, out_dir: str | Path, dt: float, camera=None, points_every: int = 10,
                 points_stride: int = 8, points_voxel: float = 0.02, every: int = 1, objects: dict | None = None,
                 save_map: bool = False):
        """camera: the renderer's Camera (width, height, fx, fy, cx, cy); default replica_pyslam().
        points_every / points_stride: the true-depth point cloud of colmap_gt back-projects every
        points_every-th saved frame at every points_stride-th pixel; points_voxel (metres) then keeps
        one point per cell of that size, so that the cloud does not grow with the length of the run (a
        Gaussian splatting trainer starts with one Gaussian per point).
        every: save every every-th step as a frame. objects: Renderer.objects(), to write objects.json
        and expect object ids in add(). save_map: write pySLAM's map to slam_state/ (large)."""
        if camera is None:
            from replica_wgpu import Camera
            camera = Camera.replica_pyslam()
        self.out = Path(out_dir)
        self.dt = dt
        self.cam = camera
        self.points_every, self.points_stride, self.points_voxel = points_every, points_stride, points_voxel
        self.every, self.objects, self.save_map = max(1, every), objects, save_map
        for d in ("results", "images", "slam"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.truth: dict[int, np.ndarray] = {}      # by step: every step
        self.estimates: dict[int, np.ndarray] = {}  # by step
        self.steps: list[int] = []                  # the step of each saved frame
        self.states: list[str] = []                 # pySLAM's tracking state at each saved frame
        self.gt_xyz, self.gt_rgb = [], []
        self.object_frames: dict[int, int] = {}     # object id -> saved frames showing it
        self.object_pixels: dict[int, int] = {}     # object id -> pixels showing it, over those frames

    def wants(self, i: int) -> bool:
        """Whether step i is saved as a frame (and so needs its object ids)."""
        return i % self.every == 0

    def add(self, i: int, rgb: np.ndarray, depth: np.ndarray, T_world_camera: np.ndarray, state: str = "",
            object_ids: np.ndarray | None = None) -> None:
        """Step i: RGB (HxWx3 uint8), depth (HxW float32 metres), its true camera-to-world pose, pySLAM's
        tracking state, and (with objects, when wants(i)) the ids of Renderer.render_objects()."""
        self.truth[i] = np.asarray(T_world_camera, np.float64)
        if not self.wants(i):
            return
        k = len(self.steps)
        self.steps.append(i)
        self.states.append(state)
        name = f"frame{k:06d}.jpg"
        jpg = self.out / "results" / name
        Image.fromarray(rgb).save(jpg, quality=95)
        link_or_copy(jpg, self.out / "images" / name)
        depth16 = np.clip(depth * DEPTH_SCALE + 0.5, 0, 65535).astype(np.uint16)
        Image.fromarray(depth16).save(self.out / "results" / f"depth{k:06d}.png")
        if self.objects is not None:
            if object_ids is None:
                raise ValueError("Exporter(objects=...) needs the object ids of every saved frame")
            Image.fromarray((object_ids + 1).astype(np.uint16)).save(self.out / "results" / f"object{k:06d}.png")
            counts = np.bincount(object_ids[object_ids >= 0].ravel())
            for oid in np.flatnonzero(counts):
                self.object_frames[int(oid)] = self.object_frames.get(int(oid), 0) + 1
                self.object_pixels[int(oid)] = self.object_pixels.get(int(oid), 0) + int(counts[oid])
        if k % self.points_every == 0:
            self._backproject(rgb, depth, T_world_camera)

    def _backproject(self, rgb, depth, T):
        s = self.points_stride
        v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
        z = depth[v, u]
        ok = (z > 0.05) & (z < 10.0)
        u, v, z = u[ok], v[ok], z[ok]
        p_c = np.stack([(u - self.cam.cx) / self.cam.fx * z, (v - self.cam.cy) / self.cam.fy * z, z], axis=1)
        self.gt_xyz.append(p_c @ T[:3, :3].T + T[:3, 3])
        self.gt_rgb.append(rgb[v, u])

    def finish(self, slam=None, run_info: dict | None = None) -> None:
        """Write the poses, pySLAM's results (if slam is given), the COLMAP and nerfstudio files, the
        objects, and run.json with run_info (e.g. the command-line arguments) in it."""
        ids = sorted(self.truth)
        saved = {k: self.truth[i] for k, i in enumerate(self.steps)}  # by frame
        with open(self.out / "traj.txt", "w") as f:
            for k in sorted(saved):
                f.write(" ".join(f"{x:.9f}" for x in saved[k].reshape(-1)) + "\n")
        with open(self.out / "groundtruth.txt", "w") as f:
            f.write("# true camera-to-world poses: t x y z qx qy qz qw\n")
            for i in ids:
                f.write(tum_line(i * self.dt, self.truth[i]) + "\n")
        with open(self.out / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "step", "t", "state"])
            w.writerows((k, i, f"{i * self.dt:.6f}", s) for k, (i, s) in enumerate(zip(self.steps, self.states)))
        if self.objects is not None:
            self._objects()
        c = self.cam
        (self.out / "cam_params.json").write_text(json.dumps({"camera": {
            "w": c.width, "h": c.height, "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "scale": DEPTH_SCALE}},
            indent=2))
        gt_xyz = np.concatenate(self.gt_xyz) if self.gt_xyz else np.zeros((0, 3))
        gt_rgb = np.concatenate(self.gt_rgb) if self.gt_rgb else np.zeros((0, 3), np.uint8)
        gt_xyz, gt_rgb = voxel_downsample(gt_xyz, gt_rgb, self.points_voxel)
        self._colmap("colmap_gt", saved, gt_xyz, gt_rgb)
        self._transforms("transforms_gt.json", saved)
        summary = {"steps": len(ids), "frames": len(self.steps), "every": self.every, "dt": self.dt,
                   "true_points": len(gt_xyz)}
        if self.objects is not None:
            summary["objects_seen"] = len(self.object_frames.keys() & self.objects.keys())
        if slam is not None:
            self._slam(slam, ids)
            if self.save_map:
                t0 = time.perf_counter()
                slam.save_system_state(str(self.out / "slam_state"))
                summary["slam_state_seconds"] = round(time.perf_counter() - t0, 1)
        root = Path(__file__).resolve().parents[1]
        try:  # the pySLAM actually loaded (it need not be the checkout next to sim/)
            import pyslam
            pyslam_dir = Path(pyslam.__file__).resolve().parents[1]
        except ImportError:
            pyslam_dir = root / "pyslam"
        versions = {"pyslam-replica-sim": git_version(root), "pyslam": git_version(pyslam_dir),
                    "replica": git_version(root / "replica")}
        run = {"run": run_info or {}, "versions": versions, "summary": summary}
        (self.out / "run.json").write_text(json.dumps(run, indent=2, default=str))

    def _objects(self):
        """objects.json: the scene's annotated objects in the true world (Replica's mesh coordinates, which
        are the world of every true pose here), with how often the saved frames showed each."""
        out = []
        for oid in sorted(self.objects):
            o = self.objects[oid]
            out.append({"id": int(oid), "class": o.class_name,
                        "center": np.round(o.center, 6).tolist(), "sizes": np.round(o.sizes, 6).tolist(),
                        "rotation": np.round(o.rotation, 9).tolist(),
                        "corners": np.round(box_corners(o.center, o.sizes, o.rotation), 6).tolist(),
                        "volume": round(o.volume, 6),
                        "frames_seen": self.object_frames.get(int(oid), 0),
                        "pixels_seen": self.object_pixels.get(int(oid), 0)})
        note = ("Boxes in the true world: corner = center + rotation @ (signs * sizes / 2). Ids are those of "
                "results/object*.png minus 1; faces of the mesh with no annotation have id 0, which is not listed. "
                "frames_seen / pixels_seen count the saved frames only.")
        (self.out / "objects.json").write_text(json.dumps({"note": note, "objects": out}, indent=1))

    def _index(self, timestamp: float) -> int:
        return int(round(timestamp / self.dt))

    def _slam(self, slam, ids):
        T_world_slam = self.truth[ids[0]]  # pySLAM's world is its first camera
        np.savetxt(self.out / "slam" / "T_world_slam.txt", T_world_slam, fmt="%.9f")
        with open(self.out / "slam" / "trajectory_online.txt", "w") as f:
            f.write("# pySLAM's camera-to-world pose of each tracked frame as tracked, in the true world\n")
            for i in sorted(self.estimates):
                f.write(tum_line(i * self.dt, T_world_slam @ self.estimates[i]) + "\n")
        poses, timestamps, _ = slam.get_final_trajectory()
        final = {self._index(t): T_world_slam @ np.asarray(T, np.float64) for T, t in zip(poses, timestamps)}
        with open(self.out / "slam" / "trajectory_final.txt", "w") as f:
            f.write("# pySLAM's final camera-to-world pose of each tracked step, in the true world\n")
            for i in sorted(final):
                f.write(tum_line(i * self.dt, final[i]) + "\n")
        frame_of = {i: k for k, i in enumerate(self.steps)}
        final = {frame_of[i]: T for i, T in final.items() if i in frame_of}  # the saved frames, by frame
        with open(self.out / "slam" / "keyframes.txt", "w") as f:
            f.write("# pySLAM's keyframes: step, t, x y z qx qy qz qw (camera-to-world, true world)\n")
            for kf in sorted((k for k in slam.map.get_keyframes() if not k.is_bad()), key=lambda k: k.timestamp):
                T = T_world_slam @ np.asarray(kf.Twc(), np.float64)
                f.write(f"{self._index(kf.timestamp)} {tum_line(kf.timestamp, T)}\n")
        points = [p for p in slam.map.get_points() if not p.is_bad()]
        xyz = np.array([np.asarray(p.pt(), np.float64).reshape(3) for p in points]).reshape(-1, 3)
        xyz = xyz @ T_world_slam[:3, :3].T + T_world_slam[:3, 3]
        bgr = np.array([np.asarray(p.color).reshape(-1)[:3] for p in points]).reshape(-1, 3)
        rgb = np.clip(bgr[:, ::-1], 0, 255).astype(np.uint8)  # pySLAM's colours are BGR, like its images
        write_ply(self.out / "slam" / "map_points.ply", xyz, rgb)
        self._colmap("colmap_slam", final, xyz, rgb)
        self._transforms("transforms_slam.json", final)
        info = {"tracked_frames_saved": len(final), "keyframes": slam.map.num_keyframes(), "map_points": len(xyz)}
        (self.out / "slam" / "info.json").write_text(json.dumps(info, indent=2))

    def add_estimate(self, i: int, T_slam_camera: np.ndarray | None) -> None:
        """pySLAM's camera-to-world pose of frame i as tracked (in pySLAM's world), or None."""
        if T_slam_camera is not None and np.isfinite(T_slam_camera).all():
            self.estimates[i] = np.asarray(T_slam_camera, np.float64)

    def _colmap(self, name, poses, xyz, rgb):
        """COLMAP model, as text and binary (gsplat's reader takes only the binary one), PINHOLE camera. COLMAP's pixel centres are at +0.5, OpenCV's at 0, so
        cx and cy grow by half a pixel. images.txt holds world-to-camera rotations (w x y z) and
        translations, each followed by an empty line of 2D points; points3D.txt has empty tracks."""
        d = self.out / name / "sparse" / "0"
        d.mkdir(parents=True, exist_ok=True)
        c = self.cam
        (d / "cameras.txt").write_text("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                                       f"1 PINHOLE {c.width} {c.height} {c.fx} {c.fy} {c.cx + 0.5} {c.cy + 0.5}\n")
        with open(d / "images.txt", "w") as f:
            f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] (empty)\n")
            for k, i in enumerate(sorted(poses), start=1):
                Tcw = np.linalg.inv(poses[i])
                w, x, y, z = quat_wxyz(Tcw[:3, :3])
                t = Tcw[:3, 3]
                f.write(f"{k} {w:.9f} {x:.9f} {y:.9f} {z:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 frame{i:06d}.jpg\n\n")
        with open(d / "points3D.txt", "w") as f:
            f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for k, (p, col) in enumerate(zip(xyz, rgb), start=1):
                f.write(f"{k} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])} 0\n")
        write_ply(d / "points3D.ply", xyz, rgb)
        write_colmap_bin(d, c, poses, xyz, rgb)
        images = self.out / name / "images"  # where gsplat and 3DGS look: hard links to results/
        images.mkdir(exist_ok=True)
        for i in poses:
            link_or_copy(self.out / "results" / f"frame{i:06d}.jpg", images / f"frame{i:06d}.jpg")

    def _transforms(self, name, poses):
        """nerfstudio transforms.json: camera-to-world in OpenGL camera axes (x right, y up, z back),
        nerfstudio's pixel-centre convention (cx + 0.5), depth PNGs with their scale."""
        c = self.cam
        frames = [{"file_path": f"results/frame{i:06d}.jpg", "depth_file_path": f"results/depth{i:06d}.png",
                   "transform_matrix": (poses[i] @ OPENGL_FROM_OPENCV).tolist()} for i in sorted(poses)]
        data = {"camera_model": "OPENCV", "w": c.width, "h": c.height, "fl_x": c.fx, "fl_y": c.fy,
                "cx": c.cx + 0.5, "cy": c.cy + 0.5, "k1": 0, "k2": 0, "p1": 0, "p2": 0,
                "depth_unit_scale_factor_note": f"depth PNG = metres x {DEPTH_SCALE}: pass "
                                                f"--depth-unit-scale-factor {1 / DEPTH_SCALE:.9f} to ns-train",
                "frames": frames}
        (self.out / name).write_text(json.dumps(data, indent=1))
