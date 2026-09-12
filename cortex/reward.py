"""
Dopamine: one scalar, broadcast to the whole sheet.

Two things it does, and they are separate:

  it teaches      nothing in the synapse knows whether the last second went
                  well. Eligibility traces remember which synapses were
                  involved; dopamine decides whether that memory becomes a
                  weight change. This is the third factor in three-factor
                  plasticity (Izhikevich 2007, "Solving the distal reward
                  problem").

  it changes now  a mouse that has just been rewarded moves differently from a
                  mouse that has not: less casting about, more commitment.
                  Tonic dopamine here scales exploration noise down and the
                  approach gain up, which is visible in the behaviour within a
                  step, long before any synapse has moved.

What it is NOT: a loss, a gradient, or a target. Nothing here backpropagates.
The brain never sees the correct answer, only whether things went better than
it had come to expect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import params


@dataclass
class Dopamine:
    """The neuromodulatory signal, plus the running expectation it fires against."""

    tonic: float = params.DA_TONIC
    tau_ms: float = params.TAU_DOPAMINE
    rpe_tau: float = params.RPE_TAU          # in control steps, not ms
    da_max: float = params.DA_MAX

    level: float = 0.0                        # current concentration
    expected: float = 0.0                     # running baseline of reward
    history: List[float] = field(default_factory=list)
    rpe_history: List[float] = field(default_factory=list)

    # ------------------------------------------------------------- signalling --
    def deliver(self, reward: float) -> float:
        """
        Hand the brain a reward and get back the surprise it caused.

        A reward you always get teaches nothing: the burst is the prediction
        error, not the reward. Deliver the same +1 often enough and the
        expectation catches up and the dopamine goes quiet, which is exactly
        what a real dopamine cell does.
        """
        reward = float(reward)
        rpe = reward - self.expected
        a = 1.0 / max(self.rpe_tau, 1.0)
        self.expected += a * rpe

        self.level = float(np.clip(self.level + rpe, -self.da_max, self.da_max))
        self.history.append(self.level)
        self.rpe_history.append(rpe)
        return rpe

    def decay(self, ms: float) -> float:
        """Let the phasic burst fall back toward tonic over `ms` milliseconds."""
        k = float(np.exp(-ms / self.tau_ms))
        self.level = self.tonic + (self.level - self.tonic) * k
        return self.level

    def reset(self) -> None:
        self.level = self.tonic
        self.expected = 0.0
        self.history.clear()
        self.rpe_history.clear()

    # -------------------------------------------------------------- behaviour --
    @property
    def arousal(self) -> float:
        """0 when starved of reward, 1 when saturated. Squashed, so one huge
        payout does not turn the animal into a different animal."""
        return float(1.0 / (1.0 + np.exp(-self.level)))

    def explore_sigma(self, base: float = 1.0) -> float:
        """Exploration noise, in units of the readout. Falls as dopamine rises:
        a rewarded animal stops casting about and commits."""
        return float(base * (1.6 - 1.2 * self.arousal))

    def approach_gain(self, base: float = params.APPROACH) -> float:
        """Fraction of the remaining gap covered per step. Rises with dopamine."""
        return float(base * (0.75 + 0.5 * self.arousal))

    def __str__(self) -> str:
        return (f"dopamine {self.level:+.3f} (expects {self.expected:+.3f}, "
                f"arousal {self.arousal:.2f})")


class RewardTrace:
    """Bookkeeping for a training run: what was paid, and whether it worked.

    Kept separate from Dopamine because one is a model of a neuromodulator and
    the other is a spreadsheet.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.rewards: List[float] = []
        self.successes: List[bool] = []

    def add(self, reward: float, success: bool = False) -> None:
        self.rewards.append(float(reward))
        self.successes.append(bool(success))

    @property
    def total(self) -> float:
        return float(np.sum(self.rewards)) if self.rewards else 0.0

    @property
    def recent(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards[-self.window:]))

    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(np.mean(self.successes[-self.window:]))

    def improved(self) -> float:
        """Mean reward in the last window minus the window before it.

        The only honest way to say a brain learned anything: it is doing better
        than an earlier version of itself under the same task.
        """
        w = self.window
        if len(self.rewards) < 2 * w:
            return 0.0
        a = np.mean(self.rewards[-2 * w:-w])
        b = np.mean(self.rewards[-w:])
        return float(b - a)

    def summary(self) -> str:
        return (f"reward total {self.total:+.2f}  recent {self.recent:+.3f}/step  "
                f"success {self.success_rate * 100:.0f}%  "
                f"delta {self.improved():+.3f}")
