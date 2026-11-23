"""Look round a Gaussian splat in a web browser: any computer, no NVIDIA GPU needed.

    pixi run python ../sim/view_splat.py $E/splat_true/splat.ply [--export $E] [--port 8080]

then open http://localhost:8080 in a web browser. The browser draws the splat (WebGL), so this runs at the
default level, on any platform. Drag to turn the view, right-drag (or two fingers) to move it, scroll to zoom.

Reads the PLY layout of 3D Gaussian Splatting, as sim/splat.py and gsplat's simple_trainer write it. With
--export, it also shows the robot's true path (green), pySLAM's (red), and the camera of every 10th saved
frame; the panel has buttons to look from a saved frame's camera, and a slider to hide faint Gaussians (the
floaters are often faint). The colour is the view-independent part only (no spherical harmonics), so shiny
surfaces look duller than in the training views.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814  # the degree-0 spherical harmonic


def load_ply(path: Path):
    """Centres, covariances, colours (0-1) and opacities of the Gaussians of a 3DGS PLY."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    col = lambda prefix, n: np.stack([np.asarray(v[f"{prefix}{i}"], np.float32) for i in range(n)], axis=1)
    centres = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(col("scale_", 3))
    q = col("rot_", 4)  # w, x, y, z
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1).reshape(-1, 3, 3)
    M = R * scales[:, None, :]
    cov = M @ M.transpose(0, 2, 1)
    rgb = np.clip(0.5 + C0 * col("f_dc_", 3), 0, 1)
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32)))
    return centres, cov, rgb, opacity[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", help="a splat (splat.ply from sim/splat.py, or a 3DGS PLY)")
    ap.add_argument("--export", help="the export it was made from: show the paths and cameras")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max", type=int, default=2_000_000, help="show at most this many Gaussians (the most opaque)")
    args = ap.parse_args()

    import viser

    t = time.time()
    centres, cov, rgb, opacity = load_ply(Path(args.ply))
    n = len(centres)
    if n > args.max:
        keep = np.argsort(-opacity[:, 0])[: args.max]
        centres, cov, rgb, opacity = centres[keep], cov[keep], rgb[keep], opacity[keep]
    print(f"{Path(args.ply)}: {n:,} Gaussians ({len(centres):,} shown), read in {time.time() - t:.0f} s")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")  # the room's frame: z up
    splat = server.scene.add_gaussian_splats("/splat", centres, cov, rgb, opacity)

    faint = server.gui.add_slider("hide opacity below", min=0.0, max=0.9, step=0.05, initial_value=0.0)

    @faint.on_update
    def _(_):
        nonlocal splat
        k = opacity[:, 0] >= faint.value
        splat.remove()
        splat = server.scene.add_gaussian_splats("/splat", centres[k], cov[k], rgb[k], opacity[k])

    if args.export:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_data import RunData

        run = RunData(args.export)
        for which, colour in (("true", (0, 160, 0)), ("slam", (220, 0, 0))):
            poses = run.poses(which)  # {saved frame: camera-to-world}
            P = np.array([poses[k][:3, 3] for k in sorted(poses)], np.float32)
            if len(P) > 1:
                server.scene.add_spline_catmull_rom(f"/path_{which}", P, color=colour, line_width=3.0)
        true_poses = run.poses("true")
        true = [true_poses[k] for k in sorted(true_poses)]
        fov = 2 * np.arctan(run.height / (2 * run.K[1, 1]))

        def wxyz(T):
            import viser.transforms as vtf
            return vtf.SO3.from_matrix(T[:3, :3]).wxyz

        for k in range(0, len(true), 10):
            server.scene.add_camera_frustum(f"/cameras/{k}", fov=fov, aspect=run.width / run.height, scale=0.1,
                                            color=(90, 90, 90), wxyz=wxyz(true[k]), position=true[k][:3, 3])
        frame = server.gui.add_slider("saved frame", min=0, max=len(true) - 1, step=1, initial_value=0)
        go = server.gui.add_button("look from that frame's true camera")

        @go.on_click
        def _(event):
            T = true[int(frame.value)]
            event.client.camera.wxyz = wxyz(T)  # an OpenCV camera, as viser's
            event.client.camera.position = T[:3, 3]
            event.client.camera.fov = fov

    print(f"\033[92mOpen http://localhost:{args.port} in a web browser; Ctrl+C here to stop.\033[0m")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
