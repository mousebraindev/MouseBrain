"""
Learning: reward-modulated spike-timing-dependent plasticity.

The problem this solves is the distal one. A synapse that helped is not told so
at the moment it helps - the reward arrives hundreds of milliseconds later, by
which time a million other synapses have also fired. So the synapse keeps a
private note, an eligibility trace, saying "I was recently part of something",
and the note decays over about a second. Dopamine, when it comes, multiplies
every note at once. Synapses with no note do not move, whatever the reward.

    dw/dt = learning_rate * dopamine(t) * eligibility(t)
    d(eligibility)/dt = -eligibility / tau_e + STDP(pre, post)

That is the three-factor rule (Izhikevich 2007). Pre and post timing sets the
sign of the note; dopamine sets whether the note is worth anything.

Two hard rules the implementation keeps:

  Dale's law      an excitatory cell cannot learn its way into being
                  inhibitory. Weights move in magnitude only; the sign is a
                  property of the cell, set by its neurotransmitter.

  a ceiling       a synapse may not exceed W_MAX_FACTOR times the strength the
                  microscope measured for it. Without it, a long run with a
                  generous reward turns the connectome into whatever the task
                  wanted, and nothing measured survives.

Cost. Decaying 2.3 million traces every 0.2 ms step would dominate everything,
so traces are held scaled: the stored array is multiplied by a single global
factor that decays instead, which makes decay O(1). The arrays are folded back
to true values once per decision, where the cost is one pass, not four hundred.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import params
from .core import Cortex


class RewardModulatedSTDP:
    """Eligibility traces over every edge, gated by a dopamine scalar."""

    def __init__(self, cortex: Cortex,
                 tau_pre: float = params.TAU_PRE,
                 tau_post: float = params.TAU_POST,
                 tau_eligibility: float = params.TAU_ELIGIBILITY,
                 a_plus: float = params.A_PLUS,
                 a_minus: float = params.A_MINUS,
                 learning_rate: float = params.LEARNING_RATE,
                 w_max_factor: float = params.W_MAX_FACTOR,
                 homeostasis: bool = True,
                 enabled: bool = True):
        self.cx = cortex
        self.dt = cortex.p.dt
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.lr = float(learning_rate)
        self.homeostasis = bool(homeostasis)
        self.enabled = enabled

        n, m = cortex.n, cortex.weights.size
        self.x = np.zeros(n, dtype=np.float32)      # presynaptic trace, scaled
        self.y = np.zeros(n, dtype=np.float32)      # postsynaptic trace, scaled
        self.e = np.zeros(m, dtype=np.float32)      # eligibility per edge, scaled

        self.kx = 1.0                                # the scale factors
        self.ky = 1.0
        self.ke = 1.0
        self.dx = float(np.exp(-self.dt / tau_pre))
        self.dy = float(np.exp(-self.dt / tau_post))
        self.de = float(np.exp(-self.dt / tau_eligibility))

        # Dale's law and the ceiling, both read off the measured weights
        self.base_sign = np.sign(cortex.base_weights).astype(np.float32)
        self.base_sign[self.base_sign == 0] = 1.0
        self.w_max = np.abs(cortex.base_weights) * float(w_max_factor)

        # incoming edges, by postsynaptic cell. The engine stores edges by
        # presynaptic column; potentiation needs them by postsynaptic row.
        post = cortex.indices
        self._in_order = np.argsort(post, kind="stable").astype(np.int64)
        self._in_ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(post, minlength=n), out=self._in_ptr[1:])

        # Synaptic scaling (Turrigiano). Without it, a global reward signal
        # moves every eligible synapse in the same direction and the network
        # simply gets louder or quieter: that global drift swamps the specific
        # credit the eligibility traces were carrying. Holding each cell's total
        # incoming weight where the microscope found it turns learning into a
        # competition between that cell's inputs, which is the only way a
        # broadcast scalar can say anything specific.
        post = cortex.indices
        self._post = post
        self._in_total = np.bincount(post, weights=np.abs(cortex.base_weights),
                                     minlength=n).astype(np.float32)

        self.updates = 0
        self.last_delta = 0.0

    # -------------------------------------------------------------- per step --
    def observe(self, fired: np.ndarray, cortex: Cortex) -> None:
        """Called once per simulation dt with the cells that just spiked."""
        if not self.enabled:
            return

        self.kx *= self.dx
        self.ky *= self.dy
        self.ke *= self.de

        if len(fired):
            # pre spikes now, post fired earlier: depression
            out = cortex.out_slots(fired)
            if len(out):
                y_true = self.y[cortex.indices[out]] * self.ky
                self.e[out] -= (self.a_minus / self.ke) * y_true

            # post spikes now, pre fired earlier: potentiation
            inc = self._in_slots(fired)
            if len(inc):
                x_true = self.x[cortex.edge_pre[inc]] * self.kx
                self.e[inc] += (self.a_plus / self.ke) * x_true

            self.x[fired] += np.float32(1.0 / self.kx)
            self.y[fired] += np.float32(1.0 / self.ky)

        if self.ke < 1e-9 or self.kx < 1e-9 or self.ky < 1e-9:
            self.renormalise()

    def _in_slots(self, post_cells: np.ndarray) -> np.ndarray:
        starts = self._in_ptr[post_cells]
        cnt = self._in_ptr[post_cells + 1] - starts
        tot = int(cnt.sum())
        if tot == 0:
            return np.empty(0, dtype=np.int64)
        off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
        return self._in_order[off + np.arange(tot)]

    def renormalise(self) -> None:
        """Fold the scale factors back into the arrays. Exact, not approximate."""
        if self.kx != 1.0:
            self.x *= np.float32(self.kx)
            self.kx = 1.0
        if self.ky != 1.0:
            self.y *= np.float32(self.ky)
            self.ky = 1.0
        if self.ke != 1.0:
            self.e *= np.float32(self.ke)
            self.ke = 1.0

    # ---------------------------------------------------------------- reward --
    def apply(self, dopamine_level: float) -> float:
        """
        Cash the eligibility traces in at the current dopamine level.

        Returns the total absolute weight change, in mV, so a caller can see
        whether anything actually moved.
        """
        if not self.enabled or dopamine_level == 0.0:
            return 0.0
        self.renormalise()

        delta = (self.lr * float(dopamine_level)) * self.e
        if not np.any(delta):
            return 0.0

        w = self.cx.weights
        mag = w * self.base_sign          # magnitude, always >= 0 for a sane graph
        mag += delta
        np.clip(mag, 0.0, self.w_max, out=mag)

        if self.homeostasis:
            mag = self._rescale(mag)

        np.multiply(mag, self.base_sign, out=w)

        self.updates += 1
        self.last_delta = float(np.abs(delta).sum())
        return self.last_delta

    def _rescale(self, mag: np.ndarray) -> np.ndarray:
        """Put each cell's total incoming weight back where it was measured."""
        now = np.bincount(self._post, weights=mag, minlength=self.cx.n)
        scale = np.ones_like(self._in_total)
        live = now > 1e-9
        scale[live] = self._in_total[live] / now[live]
        np.clip(scale, 0.5, 2.0, out=scale)      # no cell rebuilt in one payout
        mag *= scale[self._post]
        return np.clip(mag, 0.0, self.w_max)

    def clear_traces(self) -> None:
        """Between episodes: what happened in the last one is not eligible."""
        self.x[:] = 0.0
        self.y[:] = 0.0
        self.e[:] = 0.0
        self.kx = self.ky = self.ke = 1.0

    # ------------------------------------------------------------------ info --
    def stats(self) -> dict:
        self.renormalise()
        live = np.count_nonzero(self.e)
        return {
            "eligible_edges": int(live),
            "eligible_fraction": float(live / max(self.e.size, 1)),
            "eligibility_abs_mean": float(np.abs(self.e).mean()),
            "updates": self.updates,
            "last_delta_mv": self.last_delta,
        }


class Frozen:
    """A no-op stand-in, for running a trained brain without letting it change.

    Passing None to Cortex.run does the same thing; this exists so a caller can
    say `plasticity = Frozen()` and keep the rest of the code identical.
    """

    enabled = False

    def observe(self, fired, cortex) -> None:
        pass

    def apply(self, dopamine_level: float) -> float:
        return 0.0

    def _rescale(self, mag: np.ndarray) -> np.ndarray:
        """Put each cell's total incoming weight back where it was measured."""
        now = np.bincount(self._post, weights=mag, minlength=self.cx.n)
        scale = np.ones_like(self._in_total)
        live = now > 1e-9
        scale[live] = self._in_total[live] / now[live]
        np.clip(scale, 0.5, 2.0, out=scale)      # no cell rebuilt in one payout
        mag *= scale[self._post]
        return np.clip(mag, 0.0, self.w_max)

    def clear_traces(self) -> None:
        pass

    def stats(self) -> dict:
        return {"eligible_edges": 0, "updates": 0, "last_delta_mv": 0.0}
