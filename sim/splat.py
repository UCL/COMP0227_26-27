"""Gaussian splatting of a simulator run, with the true poses or with pySLAM's (Lab 06).

    pixi run -e full python ../sim/splat.py EXPORT --poses true     # trains, then scores held-out views
    pixi run -e full python ../sim/splat.py EXPORT --poses slam
    pixi run -e full python ../sim/splat.py EXPORT --load OUT/splat.ply --poses slam   # score a splat again

EXPORT is a folder written by `explore_demo.py --export EXPORT`. A splat is trained on the export's frames
with one of its two sets of poses (both in the true world: see export.py), from Gaussians started at the
points of the frames' depth images. By default only the frames that pySLAM tracked are used, with either
set of poses, so that the two splats see the same images and differ only in their poses. Every 8th of
those frames is held out: it is never trained on, and the splat is scored on it (PSNR and SSIM of the
colour, and the error of the depth the splat renders, against the simulator's depth).

Writes, in OUT (default EXPORT/splat_<poses>):
    splat.ply       the trained Gaussians, in the PLY layout of 3D Gaussian Splatting (any splat viewer
                    opens it, e.g. SuperSplat in a web browser)
    test_*.jpg      held-out views: the frame | the splat's render | the splat's depth | the depth error
    metrics.json    the scores and the settings

Needs an NVIDIA GPU (gsplat's CUDA kernels), at pySLAM's full level on Linux or WSL2. A window shows a
held-out view as training goes; press q in it to stop training early, or run with --headless.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_data import RunData, backproject  # noqa: E402

GREEN, RESET = "\033[92m", "\033[0m"


# ---------------------------------------------------------------------------------------------
# The Gaussians
# ---------------------------------------------------------------------------------------------

SH_C0 = 0.28209479177387814  # the constant spherical harmonic: colour = SH_C0 * sh0 + 0.5


def initial_points(run: RunData, poses: dict, frames: list[int], voxel: float, max_points: int, rng):
    """Points from the training frames' depth images and poses, one per voxel, with colours."""
    pts, cols = [], []
    for k in frames:
        P, vu = backproject(run.depth(k), run.K, poses[k], stride=4)
        pts.append(P)
        cols.append(run.rgb(k)[vu[:, 0], vu[:, 1]])
    P, C = np.concatenate(pts), np.concatenate(cols)
    _, keep = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
    if len(keep) > max_points:
        keep = rng.choice(keep, max_points, replace=False)
    return P[keep].astype(np.float32), C[keep].astype(np.float32) / 255.0


def make_splats(points, colours, sh_degree, device):
    import torch
    from scipy.spatial import cKDTree

    n = len(points)
    d, _ = cKDTree(points).query(points, k=4)
    dist = np.sqrt((d[:, 1:] ** 2).mean(1)).clip(1e-4, 0.1)  # the mean distance to the 3 nearest
    params = {
        "means": torch.tensor(points),
        "scales": torch.tensor(np.log(dist))[:, None].repeat(1, 3),
        "quats": torch.nn.functional.normalize(torch.rand(n, 4), dim=-1),
        "opacities": torch.logit(torch.full((n,), 0.1)),
        "sh0": torch.tensor((colours - 0.5) / SH_C0)[:, None, :],
        "shN": torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3),
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().to(device)) for k, v in params.items()})


def render(splats, viewmats, K, w, h, sh_degree):
    """Colour (H x W x 3) and expected depth (H x W) of each view, and gsplat's info."""
    import torch
    from gsplat import rasterization

    colours = torch.cat([splats["sh0"], splats["shN"]], 1)
    out, alpha, info = rasterization(
        splats["means"], splats["quats"], torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"]),
        colours, viewmats, K[None].expand(len(viewmats), 3, 3), w, h, sh_degree=sh_degree,
        render_mode="RGB+ED", packed=True)
    return out[..., :3].clamp(0, 1), out[..., 3], alpha[..., 0], info


# ---------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------

def ssim(a, b):
    """SSIM of two images (N x 3 x H x W, values 0-1), with an 11-pixel Gaussian window."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum())
    win = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x, win, padding=5, groups=3)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    s_aa, s_bb, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))).mean()


def colour_depth(d, lo=0.0, hi=5.0):
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    out = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[d <= 0] = 0
    return out


def colour_error(e, hi=0.2):
    """Depth error: white where right, red where the splat is too far, blue too near; black no depth."""
    x = np.clip(np.abs(e) / hi, 0, 1)
    out = np.full(e.shape + (3,), 255, np.uint8)
    far, near = e > 0, e < 0
    out[far, 0] = out[far, 1] = (255 * (1 - x[far])).astype(np.uint8)   # BGR: fade blue and green -> red
    out[near, 1] = out[near, 2] = (255 * (1 - x[near])).astype(np.uint8)  # fade green and red -> blue
    out[~np.isfinite(e)] = 0
    return out


def panel(rgb, pred, pred_depth, true_depth, label):
    err = np.where(true_depth > 0, pred_depth - true_depth, np.nan)
    tiles = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR),
             colour_depth(pred_depth), colour_error(err)]
    for t, name in zip(tiles, ["frame", "splat", "splat depth", "depth error (white: right)"]):
        cv2.putText(t, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    top, bottom = np.hstack(tiles[:2]), np.hstack(tiles[2:])
    out = np.vstack([top, bottom])
    cv2.putText(out, label, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------------------------
# PLY (the layout of 3D Gaussian Splatting: positions, normals, colour SH, opacity, scales, rotations)
# ---------------------------------------------------------------------------------------------

def save_ply(splats, path: Path):
    s = {k: v.detach().cpu().numpy() for k, v in splats.items()}
    n = len(s["means"])
    rest = s["shN"].transpose(0, 2, 1).reshape(n, -1)  # channel by channel, as 3DGS stores it
    cols = ([("x", s["means"][:, 0]), ("y", s["means"][:, 1]), ("z", s["means"][:, 2])]
            + [(f"n{a}", np.zeros(n)) for a in "xyz"]
            + [(f"f_dc_{i}", s["sh0"][:, 0, i]) for i in range(3)]
            + [(f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1])]
            + [("opacity", s["opacities"])]
            + [(f"scale_{i}", s["scales"][:, i]) for i in range(3)]
            + [(f"rot_{i}", s["quats"][:, i]) for i in range(4)])
    data = np.empty(n, dtype=[(name, "<f4") for name, _ in cols])
    for name, v in cols:
        data[name] = v
    header = (["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
              + [f"property float {name}" for name, _ in cols] + ["end_header"])
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(data.tobytes())


def load_ply(path: Path, device):
    import torch

    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
        data = np.frombuffer(f.read(), dtype=[(m, "<f4") for m in names], count=n)
    col = lambda prefix: np.stack([data[m] for m in names if m.startswith(prefix)], 1)  # noqa: E731
    rest = col("f_rest_")
    k = rest.shape[1] // 3
    params = {
        "means": np.stack([data["x"], data["y"], data["z"]], 1),
        "scales": col("scale_"), "quats": col("rot_"), "opacities": data["opacity"].copy(),
        "sh0": col("f_dc_")[:, None, :], "shN": rest.reshape(n, 3, k).transpose(0, 2, 1),
    }
    sh_degree = int(round(math.sqrt(k + 1))) - 1
    splats = torch.nn.ParameterDict({m: torch.nn.Parameter(torch.tensor(np.ascontiguousarray(v)).float().to(device))
                                     for m, v in params.items()})
    return splats, sh_degree


# ---------------------------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="a folder written by explore_demo.py --export")
    ap.add_argument("--poses", choices=("true", "slam"), default="true",
                    help="train (and score) with the true poses or with pySLAM's")
    ap.add_argument("--all-frames", action="store_true",
                    help="with --poses true: use every saved frame, also those pySLAM did not track")
    ap.add_argument("--steps", type=int, default=7000, help="training steps (one image each)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="image scale, e.g. 0.5 for half resolution (4x faster, less memory)")
    ap.add_argument("--sh-degree", type=int, default=3, help="degree of the view-dependent colour (0: none)")
    ap.add_argument("--test-every", type=int, default=8, help="hold out every N-th frame")
    ap.add_argument("--load", help="score this splat.ply instead of training one")
    ap.add_argument("--out", help="output folder (default EXPORT/splat_<poses>)")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("splat.py needs an NVIDIA GPU (gsplat's CUDA kernels); torch sees none. Use the provided "
                 "splats instead (the sheet says where), and look at them in a splat viewer.")
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run = RunData(args.export)
    poses = run.poses(args.poses)
    if args.all_frames and args.poses == "true":
        frames = sorted(poses)
    else:
        frames = sorted(set(run.slam_poses) & set(run.true_poses))  # what pySLAM tracked, for both
    if len(frames) < 2 * args.test_every:
        sys.exit(f"only {len(frames)} frames with {args.poses} poses: too few to train on")
    test = frames[::args.test_every]
    train = [k for k in frames if k not in set(test)]
    out = Path(args.out) if args.out else Path(args.export) / f"splat_{args.poses}"
    out.mkdir(parents=True, exist_ok=True)

    w, h = round(run.width * args.scale), round(run.height * args.scale)
    K = torch.tensor(run.K * np.array([[args.scale], [args.scale], [1]]), dtype=torch.float32, device=device)

    def load_images(ks):
        ims = [run.rgb(k) for k in ks]
        if args.scale != 1:
            ims = [cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in ims]
        return torch.tensor(np.stack(ims), device=device)  # uint8 on the GPU: 0.8 MB a frame at full size

    def viewmat(k):
        return torch.tensor(np.linalg.inv(poses[k]), dtype=torch.float32, device=device)

    print(f"{len(frames)} frames with {args.poses} poses: {len(train)} to train on, {len(test)} held out; "
          f"{w}x{h}")
    t0 = time.time()
    if args.load:
        splats, sh_degree = load_ply(Path(args.load), device)
        steps_done = 0
        print(f"loaded {len(splats['means']):,} Gaussians from {args.load}")
    else:
        from gsplat.strategy import DefaultStrategy

        pts, cols = initial_points(run, poses, train, voxel=0.02, max_points=300_000, rng=rng)
        sh_degree = args.sh_degree
        splats = make_splats(pts, cols, sh_degree, device)
        centres = np.stack([poses[k][:3, 3] for k in train])
        scene_scale = 1.1 * float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
        print(f"{len(pts):,} Gaussians to start, from the depth images; scene scale {scene_scale:.2f} m")
        lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
               "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        optimizers = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
                      for k, lr in lrs.items()}
        means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / args.steps))
        strategy = DefaultStrategy(verbose=False)
        strategy.check_sanity(splats, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)

        images = load_images(train)
        view_train = torch.stack([viewmat(k) for k in train])
        show_k = test[len(test) // 2]
        show_view, show_image = viewmat(show_k)[None], load_images([show_k])[0].cpu().numpy()
        steps_done, losses = 0, []
        for step in range(args.steps):
            i = int(rng.integers(len(train)))
            gt = images[i].float() / 255
            degree = min(step // 1000, sh_degree)  # one more SH band every 1000 steps
            colour, _, _, info = render(splats, view_train[i:i + 1], K, w, h, degree)
            strategy.step_pre_backward(splats, optimizers, state, step, info)
            a, b = colour.permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2)
            loss = 0.8 * (colour[0] - gt).abs().mean() + 0.2 * (1 - ssim(a, b))
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            means_decay.step()
            strategy.step_post_backward(splats, optimizers, state, step, info, packed=True)
            losses.append(loss.item())
            steps_done = step + 1
            if step % 100 == 0 or step == args.steps - 1:
                n = len(splats["means"])
                print(f"step {step:5d}  loss {np.mean(losses[-100:]):.4f}  Gaussians {n:,}  "
                      f"{time.time() - t0:.0f} s", flush=True)
                if not args.headless:
                    with torch.no_grad():
                        c, d, _, _ = render(splats, show_view, K, w, h, degree)
                    pan = np.hstack([cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR),
                                     cv2.cvtColor((c[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)])
                    cv2.putText(pan, f"held-out frame {show_k} | the splat, step {step}, {n:,} Gaussians", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * max(args.scale, 0.5), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("Splat (q: stop training)", pan if pan.shape[1] <= 1600 else
                               cv2.resize(pan, None, fx=1600 / pan.shape[1], fy=1600 / pan.shape[1]))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        print("stopped early (q)")
                        break
        del images
        save_ply(splats, out / "splat.ply")
    train_time = time.time() - t0

    # Score the held-out frames, each rendered from its own pose of the same set
    psnrs, ssims, depth_err, within = [], [], [], []
    with torch.no_grad():
        for j, k in enumerate(test):
            gt = load_images([k])[0].float() / 255
            c, d, _, _ = render(splats, viewmat(k)[None], K, w, h, sh_degree)
            mse = float(((c[0] - gt) ** 2).mean())
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
            ssims.append(float(ssim(c.permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))))
            true_d = run.depth(k)
            if args.scale != 1:
                true_d = cv2.resize(true_d, (w, h), interpolation=cv2.INTER_NEAREST)
            pd = d[0].cpu().numpy()
            ok = true_d > 0
            e = np.abs(pd[ok] - true_d[ok])
            depth_err.append(float(np.median(e)))
            within.append(float((e < 0.05).mean()))
            if j % max(1, len(test) // 4) == 0:
                img = panel((gt.cpu().numpy() * 255).astype(np.uint8), (c[0].cpu().numpy() * 255).astype(np.uint8),
                            pd, true_d, f"frame {k}, {args.poses} poses: PSNR {psnrs[-1]:.1f} dB, "
                                        f"depth error (median) {100 * depth_err[-1]:.1f} cm")
                cv2.imwrite(str(out / f"test_{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not args.headless:
                    cv2.imshow("Held-out view (any key: next)", cv2.resize(img, None, fx=0.6, fy=0.6))
                    cv2.waitKey(0 if args.load else 1500)
    if not args.headless:
        cv2.destroyAllWindows()

    metrics = {
        "export": str(Path(args.export).resolve()), "poses": args.poses, "frames": len(frames),
        "train_frames": len(train), "test_frames": len(test), "image": [w, h], "steps": steps_done,
        "gaussians": len(splats["means"]), "train_seconds": round(train_time, 1),
        "psnr": round(float(np.mean(psnrs)), 2), "ssim": round(float(np.mean(ssims)), 4),
        "depth_error_median_m": round(float(np.median(depth_err)), 4),
        "depth_within_5cm": round(float(np.mean(within)), 3), "loaded": args.load,
        "per_frame": {str(k): {"psnr": round(p, 2), "depth_error_m": round(de, 4)}
                      for k, p, de in zip(test, psnrs, depth_err)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    what = f"loaded {args.load}" if args.load else f"{steps_done} steps in {train_time / 60:.1f} min"
    print(f"{GREEN}Summary: splat, {args.poses} poses, {len(train)} frames trained, {len(test)} held out, "
          f"{what}, {metrics['gaussians']:,} Gaussians: PSNR {metrics['psnr']:.1f} dB, SSIM {metrics['ssim']:.3f}, "
          f"depth error {100 * metrics['depth_error_median_m']:.1f} cm (median), "
          f"{100 * metrics['depth_within_5cm']:.0f}% within 5 cm; saved in {out}{RESET}")


if __name__ == "__main__":
    main()
