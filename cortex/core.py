"""
The engine: leaky integrate-and-fire over a signed sparse connectome.

Every neuron is a LIF unit with identical passive parameters. A presynaptic
spike injects sign * n_synapses * MV_PER_SYNAPSE into each of its targets.
Excitation and inhibition come from the cell-type call, not from fitting.

Spike propagation is a ragged gather over the CSC columns, so the cost scales
with the number of neurons that actually fired, not with the population.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np

from .graph import Graph
from .params import Passive


@dataclass
class State:
    """Membrane potentials and refractory counters, carried between decisions.

    A brain does not reboot between frames. Without this, every control step is
    an independent draw and the readout has no memory of where it was pointing
    80 ms ago: measured direction autocorrelation at lag 1 goes from +0.38 with
    carried state to -0.076 without it.
    """
    v: np.ndarray
    refr: np.ndarray

    def copy(self) -> "State":
        return State(self.v.copy(), self.refr.copy())


class Cortex:
    """A population of spiking neurons wired by a Graph."""

    def __init__(self, graph: Graph, passive: Optional[Passive] = None,
                 gains: Optional[np.ndarray] = None):
        self.graph = graph
        self.p = passive or Passive()

        W = graph.W.tocsc()
        self.indptr = W.indptr
        self.indices = W.indices
        self.weights = W.data.astype(np.float32)                # plasticity edits this
        self.base_weights = W.data.astype(np.float32).copy()    # what was measured

        self.n = W.shape[0]
        self.sign = graph.sign
        self.decay = np.float32(np.exp(-self.p.dt / self.p.tau_m))
        self.refr_steps = int(np.ceil(self.p.refractory / self.p.dt))

        # per-neuron output gain: the E/I correction, and anything a caller
        # wants to scale on top of it
        self.gains = (np.ones(self.n, dtype=np.float32) if gains is None
                      else np.asarray(gains, dtype=np.float32))

        # Background input, per neuron, in mV per dt. Cortex does not sit at
        # rest waiting: a cell in an awake animal is held near threshold by a
        # balance of excitation and inhibition it never stops receiving, which
        # is what makes a handful of synapses enough to change whether it
        # fires. A cell at rest needs fifty coincident inputs and is deaf to
        # anything smaller.
        self.bias = np.zeros(self.n, dtype=np.float32)
        self.noise = np.zeros(self.n, dtype=np.float32)

        # which neuron is presynaptic to each edge. Plasticity needs it and
        # nothing else does, so it is built on first use.
        self._edge_pre: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ info --
    @property
    def edge_pre(self) -> np.ndarray:
        if self._edge_pre is None:
            counts = np.diff(self.indptr)
            self._edge_pre = np.repeat(np.arange(self.n, dtype=np.int32), counts)
        return self._edge_pre

    def hold_near_threshold(self, cells, fraction: float = 0.90,
                            jitter: float = 0.35) -> None:
        """
        Put these cells in the state an awake cortical neuron is actually in.

        `fraction` of the way from rest to threshold, with membrane noise on
        top, so the cell fires now and then on its own and every synapse
        arriving at it shifts that rate up or down. Without this a downstream
        population is silent whatever the network does, and no reward can teach
        anything about it - there is nothing for a synapse to take credit for.
        """
        cells = np.asarray(cells, dtype=np.int64)
        gap = (self.p.v_thresh - self.p.v_rest) * float(fraction)
        self.bias[cells] = np.float32(gap * self.p.dt / self.p.tau_m)
        self.noise[cells] = np.float32(jitter)

    def fresh_state(self) -> State:
        return State(np.full(self.n, self.p.v_rest, dtype=np.float32),
                     np.zeros(self.n, dtype=np.int32))

    # ------------------------------------------------------------------- run --
    def run(self,
            drive: Optional[Mapping[Iterable[int], object]] = None,
            steps: int = 400,
            record: Optional[Mapping[str, np.ndarray]] = None,
            state: Optional[State] = None,
            seed: int = 0,
            plasticity=None,
            spike_log: bool = False) -> dict:
        """
        Integrate for steps * dt milliseconds.

        drive       mapping of neuron index array to firing rate in Hz
        record      mapping of name to neuron index array, counted for spikes
        state       carried membrane state; None starts from rest
        plasticity  optional object whose observe(fired, cortex) runs every dt
        returns     name to spikes/neuron/sec, plus the underscore diagnostics
        """
        p = self.p
        rng = np.random.default_rng(seed)
        n = self.n

        st = self.fresh_state() if state is None else state.copy()
        v, refr = st.v, st.refr

        ext_idx, ext_p = self._external(drive, rng)
        record = record or {}
        counts = {k: np.zeros(len(s), dtype=np.int64) for k, s in record.items()}
        total_spikes = 0
        ever = np.zeros(n, dtype=bool)
        log = [] if spike_log else None

        indptr, indices, w, gains = self.indptr, self.indices, self.weights, self.gains
        thresh, rest, reset, decay = p.v_thresh, p.v_rest, p.v_reset, self.decay

        biased = np.any(self.bias != 0.0)
        noisy = np.any(self.noise != 0.0)

        for _ in range(steps):
            v = rest + (v - rest) * decay
            if biased:
                v += self.bias
            if noisy:
                v += self.noise * rng.standard_normal(n).astype(np.float32)

            if len(ext_idx):
                hit = ext_idx[rng.random(len(ext_idx)) < ext_p]
                if len(hit):
                    v[hit] = thresh + 1.0

            v[refr > 0] = reset
            fired = np.flatnonzero((v >= thresh) & (refr <= 0))

            if spike_log:
                log.append(fired.astype(np.int32))

            if len(fired):
                total_spikes += len(fired)
                ever[fired] = True
                refr[fired] = self.refr_steps
                v[fired] = reset

                g = self.out_slots(fired)
                if len(g):
                    cnt = indptr[fired + 1] - indptr[fired]
                    val = w[g] * np.repeat(gains[fired], cnt)
                    v += np.bincount(indices[g], weights=val, minlength=n).astype(np.float32)

                for name, sel in record.items():
                    counts[name] += np.isin(sel, fired)

            if plasticity is not None:
                plasticity.observe(fired, self)

            refr -= 1

        secs = steps * p.dt / 1000.0
        out = {k: c / secs for k, c in counts.items()}
        out["_total_hz"] = total_spikes / secs / n
        out["_spikes_per_sec"] = total_spikes / secs
        out["_fired"] = np.flatnonzero(ever)
        out["_mean_mv"] = float(v.mean())
        out["_state"] = State(v, refr)
        if spike_log:
            out["_spikes"] = log
        return out

    # --------------------------------------------------------------- helpers --
    def out_slots(self, fired: np.ndarray) -> np.ndarray:
        """Positions in the edge arrays of every outgoing edge of these cells."""
        starts = self.indptr[fired]
        cnt = self.indptr[fired + 1] - starts
        tot = int(cnt.sum())
        if tot == 0:
            return np.empty(0, dtype=np.int64)
        off = np.repeat(starts - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
        return off + np.arange(tot)

    def _external(self, drive, rng):
        if not drive:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        ii, pp = [], []
        for k, r in drive.items():
            a = np.asarray(k, dtype=np.int64)
            rr = np.asarray(r, dtype=np.float32)
            if rr.ndim == 0:
                rr = np.full(len(a), float(rr), dtype=np.float32)
            elif len(rr) != len(a):
                raise ValueError(f"drive rates ({len(rr)}) != neurons ({len(a)})")
            ii.append(a)
            pp.append(np.clip(rr * self.p.dt / 1000.0, 0.0, 1.0))
        return np.concatenate(ii), np.concatenate(pp).astype(np.float32)

    # --------------------------------------------------------------- weights --
    def weight_drift(self) -> dict:
        """How far learning has moved the wiring away from what was measured."""
        d = self.weights - self.base_weights
        moved = np.flatnonzero(np.abs(d) > 1e-6)
        return {
            "edges": int(self.weights.size),
            "changed": int(moved.size),
            "mean_abs_change_mv": float(np.abs(d).mean()),
            "max_abs_change_mv": float(np.abs(d).max()) if d.size else 0.0,
            "total_weight_mv": float(np.abs(self.weights).sum()),
        }

    def reset_weights(self) -> None:
        """Back to what the microscope measured."""
        self.weights[:] = self.base_weights
