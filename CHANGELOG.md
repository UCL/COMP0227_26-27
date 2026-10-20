# Changes to the course version of pySLAM

The `pyslam` folder is pinned to a tagged version of pySLAM's course branch, `comp0227-2026`. This page
lists what changed for you in each version, newest first. The versions are named after the lab they were
made for (`comp0227-2026-lab02`, ...); until 13 October they were numbered (`comp0227-2026.3`, ...), and
the old numbers are given in brackets. To update to the latest one, from the
`COMP0227_26-27` folder:

```bash
git pull
git submodule update --init --depth 1 pyslam
cd pyslam
pixi run build          # always; native Windows: pixi run -e default-win build
```

`build` brings the environment up to date and installs the new ready-made C++ modules; nothing you
installed before (models, data) is lost. **Do not skip it:** without it the C++ modules of your old
version keep running with the new code (since comp0227-2026-lab02 pySLAM warns about it).

## comp0227-2026-lab03.1 (20 October 2026, hotfix)

**Updating**
- `pixi run build` after updating, as always. Nothing compiles: the ready-made C++ modules are those of Lab 02.
- If `pixi run -e depth models-depth` ended with `4/5 components of 'depth' OK` and `Not installed: RAFT-Stereo`,
  run it again after updating: it downloads only RAFT-Stereo's weights (192 MB). RAFT-Stereo is a stereo
  network that Lab 03 does not use, so the lab itself was not affected.

**Setting up**
- A fresh Ubuntu, such as a new WSL2 install, has no `unzip`. `pixi run -e depth models-depth` then could not
  unpack RAFT-Stereo's weights (`unzip: command not found`) and reported that component as not installed, and
  `scripts/download_replica.sh` would have stopped after its 12 GB download. Both now unpack the zip with
  Python when there is no `unzip`. Nothing else changed since comp0227-2026-lab03.

## comp0227-2026-lab03 (20 October 2026)

**Updating**
- `pixi run build` after updating, as always. Nothing compiles: the ready-made C++ modules are those of Lab 02.

**Lab 03: depth in images and in SLAM**
- `pixi run -e depth depth-prediction` predicts the depth of one image (`--test kitti`, `--image FILE`;
  `--method DEPTH_ANYTHING_V2 | DEPTH_ANYTHING_V3 | DEPTH_PRO`, `--scene indoor|outdoor`, `--size`) or of a
  whole sequence (`--tum`, `--video`, `--images`), scores it against the sequence's own depth images
  (AbsRel, delta < 1.25, median scale; as the network gives it and with its scale removed), and saves the
  maps for SLAM (`--save NAME`, read back with `--load NAME`). `--keypoints METHOD` reads the depth at the
  ORB keypoints through `pyslam/depth_estimation/depth_sampling.py`, the function you complete in Activity 6.
  `--list` lists the methods. `--load` runs at the default level, also on native Windows; the networks need
  the `depth` level (`pixi run -e depth models-depth`, about 7 GB of weights; `models-depth-small` installs
  only the small Depth Anything V2 models, 0.2 GB).
- `pixi run slam-evaluation --predicted-depth LIST` runs monocular, RGB-D and pseudo-RGB-D SLAM on a TUM
  sequence in one report; `pixi run python scripts/trajectory_errors.py RESULTS` recomputes every run's
  trajectory error without and with a fitted scale, and prints the scale.
- `--associations` and `--predicted-depth` accept the name of the list or a path inside the sequence folder.
  `--mask-classes "tv monitor"` (a class name with a space) now reaches SLAM intact.

**Depth networks in metres**
- Depth Anything V2's metric models gave depths 2x too short indoors and 1.6x too short outdoors (the wrong
  maximum depth), Depth Anything 3's metric scaling used the wrong focal length, and both were given BGR images as if
  RGB: all fixed. Depth Pro runs on a Mac's GPU (2 s per image instead of 37 s) and reports the focal length it
  used or estimated.

**Running**
- `slam-evaluation` stops a run that has frozen (the rare hang in docs/TROUBLESHOOTING.md) and runs it
  again once: a run that has had no new frame, written no file and used almost no CPU for 100 times its
  mean time per frame (at least a minute). The frozen run's log is kept as `logs/log_stalled_1.txt` in its
  folder, with every thread's stack on Linux and macOS: please send it to the course team.
  `--stall-factor N` changes the 100; `--stall-factor 0` turns the check off.

## comp0227-2026-lab02 (12 October 2026, updated 13 October; was comp0227-2026.3)

**Updating**
- pySLAM now prints a yellow warning, and `pixi run doctor` a FAIL, when its C++ modules are older than the
  code: run `pixi run build`. On a Mac, modules left over from before comp0227-2026-lab01 made SLAM
  spend about 7 minutes loading its vocabulary before the first frame, which looked like a hang.
- macOS: `pixi run doctor` finds the files of `.pixi` marked hidden, which stops windows from opening
  (`Could not find the Qt platform plugin "cocoa"`), and gives the fix: `chflags -R nohidden .pixi`.

**Lab 02: semantic segmentation**
- `pixi run semantic-segmentation` segments one image (`--test kitti`, `--image FILE`) or a sequence
  (`--video`, `--images`, `--tum`) and can save the label maps (`--save DIR`). DeepLabV3 runs at the
  default level; SegFormer, YOLO, RF-DETR, CLIP and Detic need the semantics level
  (`pixi run -e semantics ...`). `--list` lists the methods.
- `pixi run slam --semantic-mask DIR` keeps the detector off the masked classes (people and vehicles by
  default; `--mask-classes`, `--mask-dilation`, `--mask-after-init`), also on the right image of a stereo
  pair. `pixi run slam-evaluation --semantic-mask DIR` compares SLAM with and without the mask.

**Running**
- Monocular SLAM: the final trajectory's frames now follow the map's later scale changes (after a loop
  closure), as their keyframes do. The trajectory as ORB-SLAM2/3 compute it is saved next to it
  (`trajectory_final_as_tracked.txt`) with its error, to compare. Stereo and RGB-D are unchanged.
- Loop closing: when a new loop came while the global bundle adjustment of the previous one was still
  running, no later global bundle adjustment reached the map (Linux). Fixed.
- `slam-evaluation` finds KITTI sequences under `<dataset_base_path>/sequences/`, as SLAM does.
- `slam-evaluation` puts the report, the tables and the runs straight into `results/eval_<date>/`, as the
  lab sheets say (they were in a subfolder such as `video/` or `tum/`).
- The end of a run prints the share of the frames tracked next to the trajectory error, and warns when
  tracking was lost for more than 10% of the sequence: the error covers only the tracked frames, so a run
  that lost track early can show a small one. A run in which SLAM never initialised says so, instead of
  ending in a Python error.
- `report.html` says what each table is (ATE RMSE and max in metres, tracked and lost in % of the frames),
  which way is better, and shows the best value of each row in bold. On native Windows the report was not
  written at all; it is now.
- Windows PowerShell: an `associations.txt` made with `python scripts/associate.py ... > associations.txt`
  is saved as UTF-16, which SLAM could not read. It now can.
- docs/TROUBLESHOOTING.md: a rare hang part-way through a run (every thread idle), and what to do: stop it
  and run it again.

## comp0227-2026-lab01 (7 October 2026; was comp0227-2026.2)

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

## comp0227-2026-lab00 (6 October 2026; was comp0227-2026.1)

The version the course started with: installation with pixi, and ready-made C++ modules for Linux and
macOS. Compared with upstream pySLAM it fixes, among others: memory leaks (the main process grew to over
6 GB on KITTI 06), a crash from a race between keyframe updates, a deadlock in the global bundle
adjustment, loop closing with features other than ORB, the pure-Python core, and Ctrl+Z freezing a run
(it is now ignored). Settings files without the image size work, as in ORB-SLAM2.

Tracking waits while the global bundle adjustment corrects the map after a loop closure. Before, the
correction could be held up for a long time, and tracking was lost after the loop. (This fix came on the
morning of Lab 00 as comp0227-2026.1; the version before it, comp0227-2026.0, is no longer tagged.)
