"""A controller which plays a fixed list of moves: "drive forward for 4 s, then turn left for 2 s, ...".

    pixi run python ../sim/explore_demo.py --controller script --script ../configs/route_room0.txt

A script is a text file with one move per line: how long it lasts (simulated seconds), the forward speed
(m/s) and the turn rate (rad/s, positive turns left, seen from above). '#' starts a comment.

    # seconds  speed  turn
      4.0      0.3    0.0     forward 1.2 m
      2.6      0.0    0.6     turn left about 90 degrees (2.6 s x 0.6 rad/s = 1.56 rad)

The moves are played in order, by the simulated clock, whatever pySLAM or the camera reports: the robot
follows the same route on every computer and in every run. If something stops the robot (it bumps into
furniture), it stays where it is until the next move. After the last move the robot stands still, or,
with 'repeat' on a line of its own, the script starts again.
"""
from __future__ import annotations

from pathlib import Path

from robot import Command, Controller, Estimate, Observation


def read_script(path: str | Path) -> tuple[list[tuple[float, float, float]], bool]:
    moves, repeat = [], False
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower() == "repeat":
            repeat = True
            continue
        try:
            duration, v, omega = (float(x) for x in line.split()[:3])
        except ValueError:
            raise SystemExit(f"{path}, line {n}: expected 'seconds speed turn', got {line!r}")
        if duration <= 0:
            raise SystemExit(f"{path}, line {n}: a move must last more than 0 s")
        moves.append((duration, v, omega))
    if not moves:
        raise SystemExit(f"{path}: no moves")
    return moves, repeat


class Scripted(Controller):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.moves, self.repeat = read_script(path)
        self.length = sum(m[0] for m in self.moves)
        self.bumps = 0
        # saved as controller.json in an export
        self.events = {"script": str(self.path), "repeat": self.repeat,
                       "moves": [{"seconds": d, "speed": v, "turn": w} for d, v, w in self.moves]}

    def step(self, observation: Observation, estimate: Estimate) -> Command:
        self.bumps += bool(observation.bumped)
        t = observation.time
        if self.repeat:
            t %= self.length
        for duration, v, omega in self.moves:
            if t < duration - 1e-9:
                return Command(v, omega)
            t -= duration
        return Command(0.0, 0.0)

    def summary(self) -> str:
        end = "repeats" if self.repeat else "then stands still"
        return (f"script {self.path.name}: {len(self.moves)} moves, {self.length:.1f} s, {end}; "
                f"the robot was stopped by an obstacle in {self.bumps} steps")
