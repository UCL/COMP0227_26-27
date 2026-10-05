# Setting up the software for COMP0227

This page installs pySLAM's **default level** for the course with [pixi](https://pixi.sh): visual
odometry, full SLAM with classical and learned features, loop closing (also with learned place
recognition), g2o/GTSAM optimisation, dense reconstruction and the viewers. Later weeks add more
levels (depth prediction, semantic segmentation, 3D reconstruction); they are described in
[pyslam/docs/PIXI.md](https://github.com/sjulier/pyslam/blob/main/docs/PIXI.md).

pySLAM is in the folder `pyslam` of this repository, as a git submodule: the course uses one fixed
version of it, which we move forward during the term.

pixi installs everything pySLAM needs into the repository folder (`.pixi/`), from a lock file, so
everybody gets the same versions. It does not touch conda or the rest of your system, and you do not
need to install CUDA or a compiler.

> **Do the installation before the lab.** It downloads about 6 GB with an NVIDIA GPU (the environment,
> 17 GB once unpacked; about 2 GB without one) and compiles pySLAM's C++ modules, which takes from
> 15 minutes to over an hour.

## Supported systems

| system | status |
|---|---|
| **Linux** (x86-64), with an NVIDIA GPU | tested, from older GPUs (Pascal, e.g. Titan Xp) to the RTX 50 series |
| **Linux** (x86-64), without an NVIDIA GPU | tested: add `-e default-cpu` to every `pixi run` command below |
| **macOS** (Apple silicon, macOS 14 or later) | tested. Learned features use the Apple GPU |
| **Windows** | via **WSL2** only (Ubuntu inside Windows), then as Linux: tested (Windows 11, WSL2, Titan Xp), including the GPU and the windows of `pixi run slam` |

You need about **30 GB of free disk space** (less without an NVIDIA GPU) and an internet connection.
If pySLAM's C++ modules have to be compiled on your machine (see step 3), that also needs about
**12 GB of free memory** (see [Good to know](#good-to-know)).

### Windows (WSL2): give WSL more memory before your first run

WSL2 only gets **half of your computer's memory** by default, and `pixi run slam` with its windows
needs about 8 GB: on a laptop with 16 GB of memory it does not fit. The run then stops late in the
sequence with `Killed` in the terminal, or WSL closes by itself.

Create the file `C:\Users\<you>\.wslconfig` (in Notepad, for example) containing

```
[wsl2]
memory=12GB
swap=8GB
```

then run `wsl --shutdown` in PowerShell and open Ubuntu again. `free -h` should now show a total of
about 12 GB. The rule: `memory` about 4 GB less than the computer's memory (12GB for 16 GB, 28GB for
32 GB).

- With **8 GB** of memory in the computer, use the lab machines.
- If you are short of memory, `pixi run slam --headless` needs less (about 6 GB).
- Learned features need much more: see [Good to know](#good-to-know).

### With an NVIDIA GPU: `default` or `default-cpu`?

On Linux and WSL2 with an NVIDIA GPU you can still choose the CPU-only environment
(`-e default-cpu`). The choice depends on the labs, not on the machine:

| | `default` (NVIDIA GPU) | `default-cpu` |
|---|---|---|
| download | about 6 GB | about 1.6 GB |
| on disk | about 17 GB | about 7 GB |
| ORB features (`pixi run slam` as it comes) | 32 ms per frame | 29 ms per frame |
| SuperPoint | 52 ms per frame | 340 ms per frame |
| SuperPoint + LightGlue | 91 ms per frame | 446 ms per frame |

Feature extraction and matching per frame on KITTI images (1226×370, 2000 features), measured on
one machine (Titan Xp; i9-7960X with 16 threads for the CPU). ORB-based SLAM runs at the same speed
either way, so for labs with ORB features `default-cpu` saves about 4 GB of download and 10 GB of
disk. The learned features cannot keep up with the camera on the CPU (KITTI runs at 10 frames per
second): with `default-cpu` they need `--speed 0.2` or `--throttle`, and runs take about five times
longer. On a laptop with 4 to 8 cores they will be slower still.

## 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new terminal, and check with `pixi --version` (0.81 or later). You also need `git`.

## 2. Get the code

```bash
git clone https://github.com/UCL/COMP0227_26-27.git
cd COMP0227_26-27
git submodule update --init --depth 1 pyslam
cd pyslam
```

The second command downloads pySLAM, in the version the course uses, into the folder `pyslam`
(about 0.6 GB: `--depth 1` leaves out the project's history).

If you cloned the repository before reading this and the `pyslam` folder is empty, run the
`git submodule update --init --depth 1 pyslam` line from the `COMP0227_26-27` folder: a plain
`git clone` does not download pySLAM.

**Run all the commands below from the `pyslam` folder.** If you already have an older pySLAM
checkout (for example one set up with conda), do not reuse it: use this one.

When we announce a new version, update with:

```bash
cd COMP0227_26-27
git pull
git submodule update --init --depth 1 pyslam
```

and run `pixi run build` in `pyslam` again (it only does what has changed).

## 3. Build

```bash
pixi run build      # the environment, pySLAM's C++ modules and the ORB vocabulary
pixi run check      # the C++ modules load and the optimiser tests pass
pixi run models     # the recommended learned models: SuperPoint, LightGlue and CosPlace (about 0.3 GB)
```

- The first `build` downloads the environment (about 6 GB with an NVIDIA GPU): this takes 20 minutes
  or more, mostly **without any output**, which is normal.
- `build` then installs pySLAM's C++ modules ready-made when there is a prebuilt copy for your system
  (`[native bundle] installed the prebuilt prereq modules` and `... pyslam modules`, about 30 MB of
  downloads), and otherwise compiles them, which takes from 15 minutes to over an hour.
- `build` ends by checking that all native modules share one pybind11 ABI (`OK: ... module(s) share ...`).
- `check` ends with the GTSAM and g2o tests passing.
- `models` prints one line per component, `OK` with the device it ran on (`cuda`, `mps` or `cpu`),
  and the summaries `2/2 components of 'features-core' OK` and `1/1 components of 'vpr-core' OK`.
  `models` is the same as `pixi run models-features` (SuperPoint, and SuperPoint with the LightGlue
  matcher) plus `pixi run models-vpr` (CosPlace place recognition). All 19 learned features and all
  5 place recognition models come with `pixi run models-all-features` and `pixi run models-all-vpr`
  (about 3.4 GB together).

If a command stops (for example a download breaks), **run the same command again**: what is already
done is skipped, and interrupted downloads continue where they stopped. A component reported as
`UNTRIED` could not be downloaded (no network, or its server did not answer): it is not broken, run
the same command again later.

## 4. Run SLAM

```bash
pixi run slam              # main_slam.py on the bundled KITTI 06 video
pixi run slam --headless   # without windows; prints the trajectory error (ATE) at the end
```

`pixi run slam` opens an image window, a 3D viewer, plots and loop closing's debug windows (the
keyframe similarity matrix and the loop candidates). Press `q` or `Esc` in the image window to quit.

The other main scripts have tasks too (`pixi run vo`, `pixi run feature-matching`,
`pixi run map-viewer`, ...); `pixi task list` shows them all, and [PIXI.md](https://github.com/sjulier/pyslam/blob/main/docs/PIXI.md#run) explains
how to work in a pixi shell instead.

`pixi run feature-matching` matches the features of an image pair: choose them with `--features`
and the pair with `--test`, e.g. `pixi run feature-matching --features SUPERPOINT --test mars`;
`pixi run feature-matching --list` lists the features and which `pixi run models...` task installs
their models. `pixi run vo --features ORB2` chooses the features of visual odometry the same way.

`pixi run slam-evaluation` runs SLAM several times without windows and makes a table of the results:
by default 3 runs with ORB2 and 3 with ROOT_SIFT features on the bundled KITTI 06 video (about 15
minutes). It prints the tables at the end and writes them, a `report.html` and one folder per run to
`results/eval_<date>/`:

- `rmse` and `max`: the trajectory error in metres, **over the tracked frames only**;
- `percent_tracked`: the percentage of the frames that have a pose in the final trajectory. Read the
  errors with it: a run that lost track after a tenth of the sequence has a small error;
- `percent_lost`: the percentage of the frames where tracking was lost or relocalizing.

A run that fails is reported in red with the reason.
Choose the features, the number of runs and the sequence:

```bash
pixi run slam-evaluation --features ORB2 SUPERPOINT --runs 5
pixi run slam-evaluation --video my/video.mp4 --settings settings/MY_CAMERA.yaml
pixi run slam-evaluation --images my/images --pattern "*.jpg" --fps 30 --settings settings/MY_CAMERA.yaml
pixi run slam-evaluation --tum my/tum_sequence --settings settings/MY_CAMERA.yaml
```

- `--features`: the names of `pixi run feature-matching --list`, except the `LK_*` ones.
- `--video`, `--images` (a folder of images, in the order of their names) and `--tum` need `--settings`,
  the calibration of your camera: copy a file of `settings/`, e.g. `WEBCAM.yaml`, and set `Camera.fx`,
  `Camera.fy`, `Camera.cx`, `Camera.cy`, the distortion, and the image size.
- Without ground truth there is no trajectory error, only `percent_tracked` and `percent_lost`. With `--groundtruth NAME`,
  a file in the folder of the sequence with one line per frame (`timestamp x y z qx qy qz qw scale`),
  the trajectory errors are computed too.
- `--tum` reads a sequence in the layout of the TUM RGB-D datasets as those are read: the images, their
  timestamps and their order come from the list of the frames (`associations.txt`, or `rgb.txt` for a
  sequence without depth images: `timestamp rgb/<timestamp>.png` per line), and the ground truth from
  `groundtruth.txt` (`timestamp tx ty tz qx qy qz qw` per line), which a TUM sequence must have.
- `--throttle` slows the playback down when tracking gets weak, and `--speed 0.5` plays at half the
  camera's frame rate: for features that cannot keep up with the camera (e.g. LIGHTGLUE on an older
  GPU), which otherwise lose track early. The runs take longer, and are no longer in real time.
- `--jobs 2` runs two at a time: faster, but the results get worse when the machine cannot keep up.
- `pixi run slam-evaluation --help` lists all the options. The public datasets (TUM, KITTI, EuRoC)
  are evaluated with `-c pyslam/evaluation/configs/evaluation_tum.json` (and the like) after
  downloading them and setting `dataset_base_path` in that file.

`pixi run vo` (visual odometry) shows its results in the Rerun viewer. It stops at the end of the
sequence, when you close the Rerun window, or with Ctrl+C; `pixi run vo --no-rerun` uses separate
windows instead, where `q` quits.

## Good to know

- **Results vary from run to run.** SLAM runs several threads, so two runs on the same video differ.
  On KITTI 06 the trajectory error (ATE) of monocular SLAM ranged from 11 to 21 m in our tests (several machines). Compare
  methods over several runs. See
  [non-determinism](https://github.com/sjulier/pyslam/blob/main/docs/TROUBLESHOOTING.md#non-determinism-and-run-to-run-variability).
- **SLAM runs at the camera's frame rate**, also with `--headless`. `--speed 2` plays twice as fast and
  `--speed 0` as fast as possible; faster than the camera, tracking is sometimes lost at the turns.
  Use the default speed when you compare results. With the windows open, the playback slows down by
  itself when tracking gets weak (it prints `Playback speed: ...`): drawing the windows takes CPU time
  from SLAM, and on a laptop that can be enough to lose track. `--no-throttle` turns this off. With
  `--headless` it is off; if tracking is lost at the same places in every headless run, your machine
  may be too slow for the camera's frame rate: add `--throttle`.
- **Memory for a run.** On KITTI 06, `pixi run slam` with ORB features needs about 7.5 GB with its
  windows and about 6 GB with `--headless` (measured under WSL2; the memory grows during the run, and
  stops growing when the sequence ends). With learned features it needs much more:
  `pixi run slam --features SUPERPOINT` needs about 22 GB with its windows and about 10 GB with
  `--headless`. On a laptop, run the learned features with `--headless`, or on the lab machines. If
  a run stops with `Killed`, or WSL closes by itself, it ran out of memory: on Windows see
  [the WSL memory step](#windows-wsl2-give-wsl-more-memory-before-your-first-run).
- **Do not press Ctrl+Z during a run.** Under `pixi run` it suspends SLAM but not pixi, and everything
  looks frozen. To stop a run, press `q` in a window, or Ctrl+C in the terminal. (If it happened:
  close the terminal and open a new one.)
- **Memory for the build.** Compiling GTSAM needs about 12 GB of free memory: a few files of its Python
  wrapper need up to 12 GB each. The build runs as many compiler jobs as the free memory allows; set
  `PYSLAM_BUILD_JOBS=1` to build one file at a time. On **Windows**, WSL2 gets only half of the
  computer's memory by default: if the build stops with `Killed signal terminated program cc1plus`,
  close other programs, or give WSL more memory (`memory=` and `swap=` in
  `%UserProfile%\.wslconfig`, then `wsl --shutdown`) and run `pixi run build` again. Single files of
  GTSAM and of pySLAM's C++ code need up to 12 GB, so on a laptop with 16 GB of memory (WSL2 then gets about 8 GB) give
  WSL more memory or swap **before** a build from source. None of this applies when `build`
  installs the prebuilt modules.
- **Compiler warnings** such as `-Wmaybe-uninitialized` from Eigen during a build from source are
  harmless (they appear on CPUs with AVX-512).
- **WSL2 and the Rerun viewer** (`pixi run vo`): under WSL2 pySLAM starts Rerun's viewer with its
  software Vulkan renderer (`WGPU_BACKEND=vulkan`), because the default one crashes on WSLg with
  "Invalid surface". To try another renderer, set `WGPU_BACKEND` yourself before `pixi run vo`.
- **WSL2 and the 3D viewer** (`pixi run slam`): under WSL2 the 3D viewer draws with the computer's GPU
  through WSL's Direct3D 12 driver, and prints the renderer it uses (`Viewer3D: OpenGL renderer: D3D12
  (...)`). Where that driver does not work it draws in software (`llvmpipe`), which takes several CPU
  cores. Do not install NVIDIA's Linux drivers inside WSL: the GPU comes from the Windows driver. To
  choose the renderer yourself, set `GALLIUM_DRIVER` (e.g. `llvmpipe`) before `pixi run slam`.
- **WSL2: only some of the windows open**, or none: Windows' display for Linux programs (WSLg) may have
  stopped responding. Close the Ubuntu terminals, run `wsl --shutdown` in PowerShell and start again. If
  it happens again, try `QT_QPA_PLATFORM=xcb pixi run slam`.
- **Do not `pip install` into the environment** and do not run `./clean.sh` casually: it deletes the
  build folders, and rebuilding takes up to an hour.
- **If you change pySLAM's C++ code, rebuild with `pixi run build`**, never with a module's own
  `build.sh` outside pixi (the scripts stop with an error there). After a change to pySLAM's own C++
  code (`pyslam/slam/cpp`, `cpp/`, `thirdparty/orbslam2_features`), `build` keeps the prebuilt
  libraries it is built on (GTSAM, g2o, ...) and compiles only that code (about 10 minutes on a laptop). A change
  to those libraries themselves means compiling everything, from 30 minutes to over an hour. If a
  build fails or is interrupted, run `./clean.sh` and then `pixi run build`.
- **Error messages tell you what to do.** If a component is not installed or not built, pySLAM says
  which one and which command installs or builds it.
- **Not available**: SURF (non-free). The TensorFlow-based features (DELF, LF-Net, ContextDesc,
  GeoDesc) are not in the default level.

## Troubleshooting

See [pyslam/docs/TROUBLESHOOTING.md](https://github.com/sjulier/pyslam/blob/main/docs/TROUBLESHOOTING.md) and the
[good-to-know section of PIXI.md](https://github.com/sjulier/pyslam/blob/main/docs/PIXI.md#good-to-know). When asking for help, include the full
error message, the command you ran, and the output of `pixi info` and of `pixi run doctor` (one
line per check: machine, memory, environment, GPU, every native module, vocabulary).
