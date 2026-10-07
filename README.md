# COMP0227: Scene Understanding and Situational Awareness (2026-27)

This repository is the landing page for COMP0227, which is taught in the autumn of 2026.

For this module, we will be using a heavily modified version of [pySLAM](https://github.com/luigifreda/pyslam). Modifications include: resolution of multiple crashes; resolution of multiple memory leaks; fixed assertion failures in the purely python-based optimizer; use of a single version of pybind; startup times have been reduced; various improvements have been made in the graphical and console based outputs; many more machine learned features are now supported on a Mac; the latest version of GTSAM (4.3.0) is supported; more robust heuristics to automatically scale speed and for adaptive feature selection based on quality criteria. It has also been modified to take TUM-style outputs generated from COMP0222 / COMP0249. We expect to continue to modify it to incorporate algorithms such as Amb3r and 4D Gaussian splats as the module progresses.

The code is managed through pixi. This controls how much material to download at any given time, reduces build times and resolving package versioning collisions.

## Downloading and installing (cheatsheet version)

**On a laptop?** Plug it in, and turn off Low Power Mode (macOS) or Battery Saver (Windows) before
you run SLAM: with power saving on, tracking can be lost. Windows with WSL2: do the
[WSL memory step](./SETUP.md#wsl2-give-wsl-more-memory-before-your-first-run) first.

```bash
# install pixi
curl -fsSL https://pixi.sh/install.sh | sh
```

Follow the instructions to make sure that pixi is in your path (on Windows without WSL2, see
[SETUP.md](./SETUP.md#windows-native-or-wsl2): pixi is installed from PowerShell, and every
`pixi run` command takes `-e default-win`). Then run:

```bash
# get the course materials and pySLAM
git clone https://github.com/UCL/COMP0227_26-27.git
cd COMP0227_26-27
git submodule update --init --depth 1 pyslam
cd pyslam

# install and build
pixi run build      # the environment, pySLAM's C++ modules and the ORB vocabulary
pixi run check      # the C++ modules load and the optimiser tests pass
pixi run models     # the recommended learned models: SuperPoint, LightGlue and CosPlace (about 0.3 GB)
```

**Already cloned, and the `pyslam` folder is empty?** A plain `git clone` does not download pySLAM.
Run this from the `COMP0227_26-27` folder:

```bash
git submodule update --init --depth 1 pyslam
```

To install updates:

```bash
cd COMP0227_26-27
git pull
git submodule update --init --depth 1 pyslam
cd pyslam
pixi run build
```

## Install the software (detailed version)

Follow **[SETUP.md](./SETUP.md)**. Please note: it downloads several
gigabytes. Windows users: choose between native Windows and WSL2 first.

Please **do not** follow the pySLAM installation instructions; we have wrapped the installation process very differently to make it much faster and more efficient.

## What is in this repository

| | |
|---|---|
| [SETUP.md](./SETUP.md) | how to install and check the software, and what to do when something goes wrong |
| [CHANGELOG.md](./CHANGELOG.md) | what changed in each version of the course's pySLAM, and how to update |
| `pyslam/` | [pySLAM](https://github.com/sjulier/pyslam), the visual SLAM system used in the labs, as a git submodule fixed at the version the course uses. Its own documentation is in `pyslam/docs/` |
