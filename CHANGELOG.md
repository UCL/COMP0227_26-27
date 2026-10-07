# Changes to the course version of pySLAM

The `pyslam` folder is pinned to a tagged version of pySLAM's course branch, `comp0227-2026`. This page
lists what changed for you in each version, newest first. To update to the latest one, from the
`COMP0227_26-27` folder:

```bash
git pull
git submodule update --init --depth 1 pyslam
cd pyslam
pixi run build          # native Windows: pixi run -e default-win build
```

`build` brings the environment up to date and installs the new ready-made C++ modules; nothing you
installed before (models, data) is lost.

## comp0227-2026.2 (7 October 2026)

**Windows without WSL2 (experimental).** pySLAM's default level now also runs on Windows itself, on the
CPU: about 15 GB of disk, and about 4 GB of memory for `pixi run slam` with its windows. It is meant for
laptops where WSL2 does not fit, and is experimental because it has not yet run on a slow laptop with
little memory. See [SETUP.md](./SETUP.md#windows-native-or-wsl2).

**Installing**
- Ready-made C++ modules for every supported system (Linux with or without an NVIDIA GPU, macOS, Windows):
  `pixi run build` no longer compiles anything, and takes minutes instead of up to an hour.
- One ORB vocabulary file for all systems (`ORBvoc.dbow3.bin`, a 31 MB download): it loads in a fraction
  of a second, also on macOS, where the old one took minutes.
- `pixi run models...` tries a broken download again after a pause, and says clearly what failed.
- `pixi run doctor` also runs on Windows; `pixi run doctor --run` shows the progress of its test run.

**Running**
- A quieter console: tracking's step-by-step messages go to `logs/tracking.log` (`--verbose` shows them
  again). The console keeps a progress line every 100 frames, warnings and errors.
- Monocular loop closing rejects loops with an implausible scale, and tracking restarts cleanly after the
  map is corrected: fewer lost frames after a loop closure, and a lower trajectory error on long sequences.
- Less memory for the windows on macOS and Windows.
- The TensorFlow-based features (DELF, LF-Net, ContextDesc, GeoDesc, and HDC-DELF place recognition) are
  available on Linux and macOS with `pixi run models-tf`; pySLAM runs them in an environment of their own.

## comp0227-2026.1 (6 October 2026)

- Tracking waits while the global bundle adjustment corrects the map after a loop closure. Before, the
  correction could be held up for a long time, and tracking was lost after the loop.

## comp0227-2026.0 (6 October 2026)

The version the course started with: installation with pixi, and ready-made C++ modules for Linux and
macOS. Compared with upstream pySLAM it fixes, among others: memory leaks (the main process grew to over
6 GB on KITTI 06), a crash from a race between keyframe updates, a deadlock in the global bundle
adjustment, loop closing with features other than ORB, the pure-Python core, and Ctrl+Z freezing a run
(it is now ignored). Settings files without the image size work, as in ORB-SLAM2.
