"""Preview a script of moves (sim/scripted.py) on the room's floor map, without rendering or SLAM.

    pixi run python ../sim/preview_route.py ../configs/route_room0.txt [--out route.png]

Plays the script with the simulator's robot and floor map, as explore_demo.py --controller script does, and
draws the path over the floor the robot can reach (dark grey) with the robot's start (green) and end (red),
and the room's large objects. The rest of the floor is too close to furniture for the robot (light grey,
where it would fit but cannot get to; white, where it would not fit). It takes seconds, so you can check an edited route before running it: where it ends, how far it
goes, and whether it bumps into anything (the floor map stops the robot; the camera does not see what it
bumps into). It does not show what the camera sees, or what pySLAM makes of it: for that, run it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replica" / "wgpu-renderer"))
sys.path.insert(0, str(ROOT / "sim"))

from floor_map import CELL, FloorMap  # noqa: E402
from robot import Command, Robot  # noqa: E402
from scripted import read_script  # noqa: E402

# label only the large things, once each within 0.6 m: the room has dozens of books, cushions and blinds
LABEL = {"sofa", "table", "chair", "cabinet", "lamp", "stool", "plant-stand", "shelf", "tv-screen", "desk",
         "bench", "door", "window", "picture", "basket", "vase", "bed", "nightstand", "tv-stand"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", help="a script of moves")
    ap.add_argument("--scene", default=str(ROOT / "data" / "replica" / "room_0"))
    ap.add_argument("--start", default=str(ROOT / "configs" / "start_room0.txt"),
                    help="file of camera-to-world poses; the robot starts at the first, levelled")
    ap.add_argument("--height", type=float, default=1.2, help="camera height above the floor, m")
    ap.add_argument("--dt", type=float, default=1 / 25)
    ap.add_argument("--out", help="save the picture here instead of showing it")
    args = ap.parse_args()

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from replica_wgpu.scene import read_ply

    robot = Robot.from_camera_pose(np.loadtxt(args.start).reshape(-1, 4, 4)[0])
    floor = float(np.percentile(read_ply(str(Path(args.scene) / "mesh.ply"))[0][:, 2], 1))
    robot.camera_height = floor + args.height
    fm = FloorMap.from_mesh(args.scene, robot.camera_height, robot.clearance)
    robot.x, robot.y = fm.nearest_free(robot.x, robot.y)  # as explore_demo.py
    start = (robot.x, robot.y, robot.yaw)

    moves, _ = read_script(args.script)
    path, bumps, t = [(robot.x, robot.y)], [], 0.0
    for duration, v, omega in moves:
        for _ in range(int(round(duration / args.dt))):
            if robot.step(Command(v, omega), args.dt, None, fm.blocked):
                bumps.append((robot.x, robot.y))
            path.append((robot.x, robot.y))
            t += args.dt
    P = np.array(path)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ext = [fm.x0, fm.x0 + fm.main.shape[1] * CELL, fm.y0, fm.y0 + fm.main.shape[0] * CELL]
    # floor the robot can reach (dark), floor it fits on but cannot reach from the start (light),
    # and the rest (furniture, walls, or within the robot's clearance of them: white)
    shade = np.where(fm.main, 0.55, np.where(fm.free, 0.2, 0.0))
    ax.imshow(shade, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
    info = json.loads((Path(args.scene) / "habitat" / "info_semantic.json").read_text())
    placed = []
    for o in info["objects"]:
        name = o.get("class_name", "")
        if name not in LABEL:
            continue
        # the box centre is given in the box's own frame (see the README): rotate it into the world
        box = o["oriented_bbox"]
        x, y, z, w = box["orientation"]["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        c = R @ np.asarray(box["abb"]["center"]) + np.asarray(box["orientation"]["translation"])
        if any(np.hypot(*(c[:2] - q)) < (0.6 if n == name else 0.3) for n, q in placed):
            continue
        placed.append((name, c[:2]))
        colour = "tab:blue" if name in ("window", "door") else "black"
        ax.text(c[0], c[1], name, fontsize=7, color=colour, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.plot(P[:, 0], P[:, 1], "-", color="tab:red", lw=1.5)
    ax.annotate("", xy=(start[0] + 0.3 * np.cos(start[2]), start[1] + 0.3 * np.sin(start[2])), xytext=start[:2],
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
    ax.plot(*start[:2], "o", color="tab:green", ms=8, label="start")
    ax.plot(P[-1, 0], P[-1, 1], "o", color="tab:red", ms=8, label="end")
    if bumps:
        B = np.array(bumps)
        ax.plot(B[:, 0], B[:, 1], "x", color="orange", ms=8, label="bumped")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower right", fontsize=8)
    summary = (f"{Path(args.script).name}: {len(moves)} moves, {t:.1f} s ({round(t / args.dt)} steps), "
               f"travelled {robot.distance:.1f} m, bumped in {len(bumps)} steps")
    ax.set_title(summary + "\ndark grey: the floor the robot can reach; light grey: floor it fits on but cannot reach",
                 fontsize=10)
    print(f"\033[92mSummary: {summary}\033[0m")
    if args.out:
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
