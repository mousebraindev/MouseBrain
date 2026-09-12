"""
A cue, and one of k answers.

This is the discrete-choice half of the interface: the world shows a pattern,
the brain picks a bin, and it is right or it is not. Chance is 1/k, which makes
it the one task in the repo where "did it learn anything" has an unarguable
baseline.

There is nothing visual about the mapping from bin to answer - the brain is not
told that bin 2 means answer 2, and the binning of preferred directions has no
reason to line up with the cue. Whatever alignment appears has to be learned,
and if none appears the honest report is that none appeared.
"""
from __future__ import annotations

import numpy as np

from ..field import render_points
from ..readout import Action
from ..task import Task


class CueChoice(Task):
    """One cue in a quadrant, k answers, reward for the matching one."""

    name = "cue-choice"

    def __init__(self, k: int = 4, size=(120, 120), reward: float = 1.0,
                 penalty: float = 0.25, seed: int = 0):
        self.choices = int(k)
        self.w, self.h = int(size[0]), int(size[1])
        self.reward_hit = float(reward)
        self.penalty = float(penalty)
        self.rng = np.random.default_rng(seed)
        self._answered = False
        self._correct = False
        self.reset()

    def reset(self) -> None:
        self.cue = int(self.rng.integers(0, self.choices))
        self._answered = False
        self._correct = False

    def observe(self) -> np.ndarray:
        # the cue sits on a circle, one position per option, so the k cues are
        # as distinguishable as a visual area can make them
        ang = 2 * np.pi * self.cue / self.choices
        r = 0.32 * min(self.w, self.h)
        pt = np.array([[self.w / 2 + r * np.cos(ang), self.h / 2 - r * np.sin(ang)]])
        field = render_points(pt, size=(self.w, self.h), radius=7.0)
        return field

    def act(self, action: Action) -> float:
        self._answered = True
        if action.choice < 0:
            return -self.penalty * 0.5          # said nothing at all
        self._correct = (action.choice == self.cue)
        return self.reward_hit if self._correct else -self.penalty

    def done(self) -> bool:
        return self._answered

    def succeeded(self) -> bool:
        return self._correct

    @property
    def chance(self) -> float:
        """What a coin gets, for comparison. Report against this, not against 0."""
        return 1.0 / self.choices
