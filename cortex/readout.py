"""
The readout: spikes out, a decision out.

Cortex has no motor neurons. Nothing in this cubic millimetre ever moved a
mouse's paw, so there is no cell to read that means "go left". Every readout
here is a decode - the same thing a neural prosthesis does when it drives a
cursor from motor cortex - and calling it a command would be a lie.

Three of them, because different tasks want different shapes of answer:

  vector        where to move. Two decoders are available. The retinotopic one
                takes the centre of mass of the firing population in receptive
                field coordinates: activity at a place on the map means
                something is at that place, which is how a saccade is
                generated. The direction one is the classic population vector
                over preferred directions; it reports which way an edge is
                moving, not where anything is, which is why on its own it
                drifts rather than arrives.

  commit        the discrete act - a click, a grasp, a submit. The fly
                connectome has a stopping neuron and gets this for free.
                Cortex has none, so here it is the population having arrived:
                still firing hard, but no longer pulled anywhere. That is a
                modelling choice made by a person.

  choice        one of k options. A motor pool - k disjoint groups of cells
                that the eye does not drive - is read out by mean rate, and the
                loudest group wins. Nothing in the anatomy says group 2 means
                option 2: the mapping is arbitrary, and if it ever lines up
                with the task it is because reward made it line up. This is
                what makes the brain usable for anything that is not pointing
                at a place.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np

from . import params
from .field import Retina


@dataclass
class Action:
    """What the population decided, in units the caller can use."""
    dx: float = 0.0
    dy: float = 0.0
    commit: bool = False
    choice: int = -1
    confidence: float = 0.0        # 0 when the population cancels out
    hz: float = 0.0                # mean rate of the decoding population
    detail: dict = _field(default_factory=dict)

    @property
    def silent(self) -> bool:
        return self.hz < params.QUIET_HZ

    def __str__(self) -> str:
        return (f"({self.dx:+6.1f},{self.dy:+6.1f}) "
                f"{'COMMIT' if self.commit else '      '} "
                f"conf {self.confidence:.3f}  {self.hz:5.1f} Hz")



class MotorPool:
    """k disjoint groups of cells, read out by rate.

    Cortex has no motor neurons, so a discrete act has to be decoded from cells
    that were never built to produce one. The pool is drawn from cells the eye
    does not drive, so a group cannot win simply by sitting where the light is:
    whatever reaches it came through the network.

    The membership is fixed at construction and saved with nothing - it is a
    property of the decoder, like the placement of an electrode array, not
    something the brain learns.
    """

    def __init__(self, n_neurons: int, k: int = 4, exclude: Optional[np.ndarray] = None,
                 size: Optional[int] = None, seed: int = 0, W=None,
                 hold: float = 0.92, jitter: float = 0.45):
        rng = np.random.default_rng(seed)
        exclude = np.asarray([] if exclude is None else exclude, dtype=np.int64)
        free = np.setdiff1d(np.arange(n_neurons, dtype=np.int64), exclude)
        self.overlaps_input = False
        if free.size < k * 8:
            # not enough undriven cells: fall back to the whole population and
            # say so, because a pool that overlaps the eye is a weaker claim
            free = np.arange(n_neurons, dtype=np.int64)
            self.overlaps_input = True

        per = int(size or max(8, min(free.size // (2 * k), 128)))

        # An electrode is not dropped at random: it goes where there is signal.
        # Rank the free cells by how much input they receive from the eye and
        # keep the best-connected few times what is needed, then shuffle those
        # into groups - so which group a cell lands in is arbitrary, but every
        # cell in the pool is at least reachable.
        if W is not None and exclude.size:
            v = np.zeros(n_neurons, dtype=np.float32)
            v[exclude] = 1.0
            reach = np.abs(W) @ v
            order = free[np.argsort(reach[free])[::-1]]
            pool_from = order[:min(len(order), k * per * 4)]
            self.reached = float((reach[pool_from] > 0).mean())
        else:
            pool_from = free
            self.reached = float("nan")

        pick = rng.permutation(pool_from)[:k * per]
        self.k = int(k)
        self.groups = [np.sort(pick[i * per:(i + 1) * per]) for i in range(k)]
        self.index = np.concatenate(self.groups)
        self.bounds = np.cumsum([0] + [len(g) for g in self.groups])
        # Cortex idles, and a pool held at exactly zero can only be read in one
        # direction. But the idling has to come from the membrane, not from an
        # injected spike train: a cell forced to fire by something outside the
        # network fires the same whatever the network does, and no synapse can
        # earn credit for it. So the pool is held near threshold instead, and
        # what pushes it over is the input it receives.
        self.hold = float(hold)
        self.jitter = float(jitter)

    def install(self, cortex) -> None:
        """Put the pool into the near-threshold state. Called once."""
        cortex.hold_near_threshold(self.index, fraction=self.hold, jitter=self.jitter)

    def __len__(self) -> int:
        return int(self.index.size)

    def rates(self, pool_hz: np.ndarray) -> np.ndarray:
        """Mean rate per group, in group order."""
        return np.array([float(pool_hz[self.bounds[i]:self.bounds[i + 1]].mean())
                         for i in range(self.k)])

    def describe(self) -> str:
        note = " (overlapping the eye - too few free cells)" if self.overlaps_input else ""
        reach = ("" if np.isnan(self.reached)
                 else f", {self.reached * 100:.0f}% reachable from the eye")
        return f"motor pool: {self.k} groups of {len(self.groups[0])} cells{reach}{note}"


class Readout:
    """Turns recorded spike rates into an Action."""

    def __init__(self, retina: Retina,
                 mode: str = "retinotopic",
                 gdsi_min: float = params.GDSI_MIN,
                 quiet_hz: float = params.QUIET_HZ,
                 smooth: float = params.SMOOTH,
                 approach: float = params.APPROACH,
                 max_step: float = 90.0,
                 commit_below: float = 0.045,
                 commit_hz: float = 6.0,
                 pool: Optional[MotorPool] = None):
        if mode not in ("retinotopic", "direction"):
            raise ValueError("mode has to be 'retinotopic' or 'direction'")
        self.retina = retina
        self.mode = mode
        self.pool = pool
        self.quiet_hz = quiet_hz
        self.smooth = smooth
        self.approach = approach
        self.max_step = max_step
        self.commit_below = commit_below
        self.commit_hz = commit_hz

        self.all_idx = retina.idx
        self.u, self.v = retina.u, retina.v
        # The receptive fields do not tile the view evenly, so "no signal" is
        # not the middle of the sheet. Measure the resting centre of mass and
        # treat displacement from it as the thing that means something.
        self.u_mid = float(self.u.mean())
        self.v_mid = float(self.v.mean())

        m = retina.gDSI >= gdsi_min
        if not m.any():
            m = retina.gDSI >= 0.0
        self.dir_idx = retina.idx[m]
        self.dir_ang = retina.pref_dir[m]
        self.dir_w = np.maximum(retina.gDSI[m], 1e-3)
        self.ux = np.cos(self.dir_ang) * self.dir_w
        self.uy = -np.sin(self.dir_ang) * self.dir_w

        # The sample is not balanced round the clock: in the measured volume,
        # 957 cells prefer 225-270 degrees and 361 prefer 135-180. Firing every
        # cell at the same rate already yields dx=-1.9 dy=+13.5, which would be
        # reported as the cortex deciding to go down-left. Measure that offset
        # once and subtract it, so what is left is the response to the image.
        self.bias_x = float(self.ux.mean())
        self.bias_y = float(self.uy.mean())

        self._vx = self._vy = 0.0

    def populations(self) -> dict:
        """What to hand Cortex.run as its record argument."""
        out = {"all": self.all_idx, "dir": self.dir_idx}
        if self.pool is not None:
            out["pool"] = self.pool.index
        return out

    def reset(self) -> None:
        self._vx = self._vy = 0.0

    # ---------------------------------------------------------------- decode --
    def decode(self, rates: dict,
               view: tuple = (1.0, 1.0),
               approach: Optional[float] = None,
               noise: float = 0.0,
               rng: Optional[np.random.Generator] = None) -> Action:
        """
        rates     the dict Cortex.run returned
        view      (w, h) of the field the gaze covered, in caller units
        approach  fraction of the remaining gap to cover; None uses the default
        noise     exploration, in caller units, added to the vector
        """
        dir_hz = np.asarray(rates["dir"], dtype=np.float32)
        all_hz = np.asarray(rates["all"], dtype=np.float32)
        mean_hz = float(dir_hz.mean()) if dir_hz.size else 0.0
        vw, vh = float(view[0]), float(view[1])

        if mean_hz < self.quiet_hz:
            # Nothing in view. With no drive the ratios below are 0/0 and the
            # bias subtraction would report a confident push in the opposite
            # direction, so a silent population has to mean a still output.
            return Action(hz=mean_hz, detail={"reason": "silent"})

        if self.mode == "retinotopic":
            tot = float(all_hz.sum()) + 1e-9
            cu = float((all_hz * self.u).sum()) / tot
            cv = float((all_hz * self.v).sum()) / tot
            vx = (cu - self.u_mid) * vw
            vy = (cv - self.v_mid) * vh
        else:
            tot = float(dir_hz.sum()) + 1e-9
            vx = (float((dir_hz * self.ux).sum()) / tot - self.bias_x) * vw
            vy = (float((dir_hz * self.uy).sum()) / tot - self.bias_y) * vh

        # A motor system integrates too: muscle does not follow spikes one for
        # one, and a vector built from a single 80 ms window is close to a coin
        # flip because a decoder cell fires 0.32 times in 20 ms.
        a = self.smooth
        self._vx = a * self._vx + (1 - a) * vx
        self._vy = a * self._vy + (1 - a) * vy
        vx, vy = self._vx, self._vy

        if noise > 0.0:
            rng = rng or np.random.default_rng()
            vx += float(rng.normal(0.0, noise))
            vy += float(rng.normal(0.0, noise))

        k = self.approach if approach is None else float(approach)
        dx = float(np.clip(vx * k, -self.max_step, self.max_step))
        dy = float(np.clip(vy * k, -self.max_step, self.max_step))

        # Taking a fixed fraction of the gap means the step shrinks as the
        # target closes in, and the last stretch becomes a crawl: median step
        # measured at 1.2 px. Keep a floor while a real gap remains.
        mag = float(np.hypot(dx, dy))
        want = float(np.hypot(vx, vy))
        if 0.4 < mag < 7.0 and want > 6.0:
            dx, dy = dx * 7.0 / mag, dy * 7.0 / mag

        # distance still to go, normalised by the view: falls toward zero as the
        # population arrives, which is exactly when a commit should happen
        gap = float(np.hypot(vx / max(vw, 1e-9), vy / max(vh, 1e-9)))
        commit = (gap < self.commit_below) and (mean_hz >= self.commit_hz)

        return Action(dx=dx, dy=dy, commit=commit,
                      confidence=float(np.clip(1.0 - gap, 0.0, 1.0)),
                      hz=mean_hz,
                      detail={"gap": gap, "vx": vx, "vy": vy, "mode": self.mode})

    # ---------------------------------------------------------------- choose --
    def choose(self, rates: dict, k: int = 4) -> Action:
        """
        Pick one of k options by reading the motor pool.

        The loudest group wins; confidence is how far ahead it is. The mapping
        from group to meaning belongs to the task, not to the brain, and the
        brain is never told it - reward is the only thing that can align them.
        """
        if self.pool is None:
            raise ValueError(
                "this readout has no motor pool, so it cannot make a discrete "
                "choice. Build the Brain with choices=k, or pass "
                "Readout(pool=MotorPool(...)).")
        if k != self.pool.k:
            raise ValueError(f"pool has {self.pool.k} groups, task asked for {k}")

        pool_hz = np.asarray(rates["pool"], dtype=np.float32)
        mean_hz = float(pool_hz.mean()) if pool_hz.size else 0.0
        if mean_hz < self.quiet_hz:
            return Action(hz=mean_hz, detail={"reason": "silent pool"})

        score = self.pool.rates(pool_hz)
        order = np.argsort(score)[::-1]
        best = int(order[0])
        margin = float((score[order[0]] - score[order[1]]) / (score[order[0]] + 1e-9))             if k > 1 else 1.0
        return Action(choice=best, confidence=float(np.clip(margin, 0.0, 1.0)),
                      hz=mean_hz, detail={"scores": score.tolist()})
