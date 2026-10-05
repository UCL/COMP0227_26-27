# COMP0227: Scene Understanding and Situational Awareness (2026-27)

This repository is the landing page for COMP0227, which is taught in the autumn of 2026.

## Get the code

```bash
git clone https://github.com/UCL/COMP0227_26-27.git
cd COMP0227_26-27
git submodule update --init --depth 1 pyslam
```

**Already cloned, and the `pyslam` folder is empty?** A plain `git clone` does not download pySLAM.
Run the last command above from the `COMP0227_26-27` folder:

```bash
git submodule update --init --depth 1 pyslam
```

## Then install the software

Follow **[SETUP.md](./SETUP.md)**. Please do this **before** the first lab: it downloads several
gigabytes. Windows users: read the WSL memory step first.

## What is in this repository

| | |
|---|---|
| [SETUP.md](./SETUP.md) | how to install and check the software, and what to do when something goes wrong |
| `pyslam/` | [pySLAM](https://github.com/sjulier/pyslam), the visual SLAM system used in the labs, as a git submodule fixed at the version the course uses. Its own documentation is in `pyslam/docs/` |
