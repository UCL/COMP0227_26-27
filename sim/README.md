# The robot simulator

A simulated robot drives round a Replica room. The Replica renderer (`../replica/wgpu-renderer`) draws what
its camera sees, pySLAM tracks it, and a controller decides where to go next. Lab 06 introduces it; the
coursework builds on it.

Run everything from the `pyslam` folder, as for the labs (`pixi run python ../sim/...`). The scripts need
the room in `../data/replica/room_0` (see the Lab 06 sheet for the download).

| file | does |
|---|---|
| `explore_demo.py` | the simulator: robot, renderer, pySLAM and a controller in one loop; `--gui` opens the windows, `--export DIR` saves the run |
| `lockstep_demo.py` | pySLAM against the renderer along a recorded path: a quick check that both work |
| `robot.py` | the robot (a unicycle, a disc of radius 0.35 m) and the example controller, `Wander` |
| `scripted.py` | a controller that plays a file of moves (`--controller script --script FILE`) |
| `floor_map.py` | where the robot can be, from the room's mesh (the simulator's, not the robot's) |
| `preview_route.py` | draws a script's route on the floor plan, without running anything |
| `export.py`, `run_data.py` | write and read a saved run |
| `tsdf.py` | a TSDF mesh from a saved run, with the true or pySLAM's poses |
| `splat.py` | a Gaussian splat from a saved run (NVIDIA GPU, `-e full`) |
| `view_splat.py` | look round a splat in a web browser |

`../configs/` holds the room's camera settings, the start pose and the Lab 06 route.
