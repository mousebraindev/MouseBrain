"""
Reach the bright thing.

The smallest honest test of the whole loop: a field with one target and three
distractors, a cursor, and a reward for closing the distance. No text, no
semantics, nothing a visual area could not in principle do.

Reward is shaped - paid every step for the distance it just closed - because
eligibility traces reach back about a second, not across an episode. A single
payout at the end would arrive long after the synapses that earned it had
forgotten they were involved.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..field import render_points
from ..readout import Action
from ..task import Task


class ReachTarget(Task):
    """Move the cursor onto the brightest blob in the field."""

    name = "reach-target"

    def __init__(self, size: Tuple[int, int] = (160, 120),
                 distractors: int = 3,
                 radius: float = 10.0,
                 step_cost: float = 0.01,
                 seed: int = 0):
        self.w, self.h = int(size[0]), int(size[1])
        self.distractors = int(distractors)
        self.radius = float(radius)
        self.step_cost = float(step_cost)
        self.rng = np.random.default_rng(seed)
        self.diag = float(np.hypot(self.w, self.h))
        self.reset()

    # ----------------------------------------------------------------- world --
    def reset(self) -> None:
        m = 16
        self.target = np.array([self.rng.uniform(m, self.w - m),
                                self.rng.uniform(m, self.h - m)], dtype=np.float32)
        self.decoys = np.array(
            [[self.rng.uniform(m, self.w - m), self.rng.uniform(m, self.h - m)]
             for _ in range(self.distractors)], dtype=np.float32
        ).reshape(-1, 2)
        self.cursor = np.array([self.w / 2.0, self.h / 2.0], dtype=np.float32)
        self._prev = self._distance()
        self._arrived = False
        self._steps = 0

    def _distance(self) -> float:
        return float(np.linalg.norm(self.cursor - self.target))

    def observe(self) -> np.ndarray:
        pts = np.vstack([self.target[None, :], self.decoys]) if len(self.decoys) \
            else self.target[None, :]
        weights = np.concatenate([[1.0], np.full(len(self.decoys), 0.45)])
        field = render_points(pts, size=(self.w, self.h), radius=8.0, weights=weights)
        # the cursor is part of the scene: a small dim mark, so the population
        # has something local to work against
        field = np.maximum(field, 0.30 * render_points(
            self.cursor[None, :], size=(self.w, self.h), radius=3.0))
        return field

    # ------------------------------------------------------------------ act --
    def act(self, action: Action) -> float:
        self._steps += 1
        self.cursor[0] = float(np.clip(self.cursor[0] + action.dx, 0, self.w - 1))
        self.cursor[1] = float(np.clip(self.cursor[1] + action.dy, 0, self.h - 1))

        d = self._distance()
        closed = (self._prev - d) / self.diag      # fraction of the field closed
        self._prev = d

        reward = float(closed) - self.step_cost
        if d <= self.radius:
            self._arrived = True
            reward += 1.0
            if action.commit:
                reward += 0.5                      # arrived and knew it
        elif action.commit:
            reward -= 0.25                         # committed to the wrong place
        return reward

    def done(self) -> bool:
        return self._arrived

    def succeeded(self) -> bool:
        return self._arrived

    def focus(self) -> Optional[Tuple[float, float]]:
        return None      # the whole field is in view; the target may be anywhere

    def window(self) -> Optional[Tuple[float, float]]:
        return None
