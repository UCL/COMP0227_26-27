# COMP0227: Scene Understanding and Situational Awareness (2026-27)

This repository is the landing page for COMP0227, which is taught in the autumn of 2026.

## Downloading and installing (cheatsheet version)

```bash
# install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# clone the repo and active the submodule
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
Run the last command above from the `COMP0227_26-27` folder:

```bash
git submodule update --init --depth 1 pyslam
```

To install updates:

```bash
cd COMP0227_26-27
git pull
git submodule update --init --depth 1 pyslam
```

## Install the software (detailed version)

Follow **[SETUP.md](./SETUP.md)**. Please note: it downloads several
gigabytes. Windows users: read the WSL memory step first.

Please **do not** follow the pySLAM installation instructions; we have wrapped the installation process very differently to make it much faster and more efficient.

## What is in this repository

| | |
|---|---|
| [SETUP.md](./SETUP.md) | how to install and check the software, and what to do when something goes wrong |
| `pyslam/` | [pySLAM](https://github.com/sjulier/pyslam), the visual SLAM system used in the labs, as a git submodule fixed at the version the course uses. Its own documentation is in `pyslam/docs/` |
