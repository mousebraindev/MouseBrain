"""
The whole animal, in one object.

    from cortex import Brain

    brain = Brain.demo()                  # runs anywhere, no download
    act = brain.look(field)               # a 2D array in, a decision out
    brain.reward(+1.0)                    # and it learns from it

Everything below that line - the sparse matrix, the eligibility traces, the E/I
correction - is machinery. This is the surface you attach to a task.

What a Brain is: a connectome, a population of spiking cells, an eye that turns
a field into drive, a readout that turns spikes into a decision, a dopamine
signal, and the synaptic rule the dopamine gates. It carries membrane state
between decisions, because a brain does not reboot between frames.

What a Brain is not: goal-directed out of the box. Untrained, it drifts along
contrast. Reward is what gives it somewhere to go.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import params
from .balance import per_neuron_gains
from .core import Cortex
from .field import Retina
from .graph import Graph
from .plasticity import Frozen, RewardModulatedSTDP
from .readout import Action, MotorPool, Readout
from .reward import Dopamine, RewardTrace


class Brain:
    def __init__(self,
                 graph: Graph,
                 sim_steps: int = params.SIM_STEPS,
                 mode: str = "retinotopic",
                 choices: Optional[int] = None,
                 learn: bool = True,
                 explore: float = 0.0,
                 balance: bool = True,
                 seed: int = 0,
                 quiet: bool = False):
        self.graph = graph
        self.sim_steps = int(sim_steps)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        gains = per_neuron_gains(graph, verbose=not quiet) if balance else None
        self.cortex = Cortex(graph, gains=gains)
        self.retina = Retina(graph)
        pool = (MotorPool(self.cortex.n, k=choices, exclude=self.retina.idx,
                          seed=seed, W=graph.W)
                if choices else None)
        if pool is not None:
            pool.install(self.cortex)
        self.readout = Readout(self.retina, mode=mode, pool=pool)
        self.dopamine = Dopamine()
        self.trace = RewardTrace()
        self.plasticity = (RewardModulatedSTDP(self.cortex) if learn else Frozen())

        self.explore = float(explore)
        self.state = None
        self.steps = 0
        self.last: Optional[Action] = None
        self.last_rates: Optional[dict] = None

        if not quiet:
            print(graph.describe())
            print(self.retina.describe())
            if pool is not None:
                print(pool.describe())

    # --------------------------------------------------------- constructors --
    @classmethod
    def demo(cls, n: int = 4000, seed: int = 0, **kw) -> "Brain":
        """A brain with synthetic wiring. Runs in seconds, needs no download,
        and is not evidence about any mouse."""
        return cls(Graph.synthetic(n=n, seed=seed), **kw)

    @classmethod
    def measured(cls, path=None, **kw) -> "Brain":
        """The MICrONS cube, after `cortex build`."""
        g = Graph.load() if path is None else Graph.load(path)
        return cls(g, **kw)

    # ---------------------------------------------------------------- sense --
    @property
    def window_ms(self) -> float:
        """How much brain time one decision takes."""
        return self.sim_steps * self.cortex.p.dt

    def look(self,
             field: np.ndarray,
             focus: Optional[Tuple[float, float]] = None,
             window: Optional[Tuple[float, float]] = None,
             view: Optional[Tuple[float, float]] = None,
             k: Optional[int] = None) -> Action:
        """
        Show the brain a 2D field and get back what it decided.

        focus / window  where and how wide it looks, in field pixels
        view            the caller units one full gaze spans; defaults to the
                        window, so dx and dy come back in field pixels
        k               if given, decide among k options instead of a direction
        """
        arr = np.asarray(field)
        h, w = (arr.shape[0], arr.shape[1]) if arr.ndim >= 2 else (1, 1)
        win = window or (float(w), float(h))
        view = view or win

        drive = self.retina.look(field, focus=focus, window=window,
                                 gain=1.0 + 0.25 * self.dopamine.arousal)
        rates = self.cortex.run(drive,
                                steps=self.sim_steps,
                                record=self.readout.populations(),
                                state=self.state,
                                seed=int(self.rng.integers(1 << 31)),
                                plasticity=self.plasticity)
        self.state = rates["_state"]
        self.last_rates = rates
        self.steps += 1

        # dopamine falls back toward tonic over the time the decision took
        self.dopamine.decay(self.window_ms)

        if k is not None:
            act = self.readout.choose(rates, k=k)
        else:
            act = self.readout.decode(
                rates, view=view,
                approach=self.dopamine.approach_gain(),
                noise=(self.dopamine.explore_sigma(self.explore)
                       if self.explore > 0 else 0.0),
                rng=self.rng)
        act.detail["spikes_per_sec"] = float(rates["_spikes_per_sec"])
        act.detail["mean_mv"] = float(rates["_mean_mv"])
        act.detail["dopamine"] = self.dopamine.level
        self.last = act
        return act

    # --------------------------------------------------------------- reward --
    def reward(self, value: float, success: bool = False) -> float:
        """
        Pay the brain, and let the payment reach back to what caused it.

        Every synapse that was recently active holds an eligibility trace.
        Dopamine multiplies all of them at once, so the synapses that were part
        of whatever just worked move, and the rest do not. Returns the
        prediction error, which is what actually drove the change: a reward the
        brain already expected moves nothing.
        """
        rpe = self.dopamine.deliver(value)
        self.plasticity.apply(self.dopamine.level)
        self.trace.add(value, success)
        return rpe

    def punish(self, value: float = 1.0) -> float:
        """Negative reward. Weakens whatever was recently eligible."""
        return self.reward(-abs(float(value)))

    def new_episode(self) -> None:
        """Between attempts: fresh membranes, fresh traces, dopamine at tonic.

        Eligibility from the last episode must not be cashed in by this one -
        that is how a brain learns a superstition.
        """
        self.state = None
        self.readout.reset()
        self.plasticity.clear_traces()
        self.dopamine.level = self.dopamine.tonic

    # ------------------------------------------------------------ inference --
    def freeze(self) -> "Brain":
        """Stop learning. The brain still runs; nothing moves any more."""
        self.plasticity = Frozen()
        return self

    def thaw(self) -> "Brain":
        self.plasticity = RewardModulatedSTDP(self.cortex)
        return self

    # ----------------------------------------------------------------- disk --
    @property
    def fingerprint(self) -> str:
        """Identifies the connectome this brain was trained on, so a checkpoint
        cannot be loaded onto different wiring."""
        h = hashlib.sha256()
        h.update(np.asarray(self.cortex.base_weights, dtype=np.float32).tobytes())
        h.update(str(self.cortex.n).encode())
        return h.hexdigest()[:16]

    def save(self, path) -> Path:
        """Write what training changed: the weights, and nothing else."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "fingerprint": self.fingerprint,
            "neurons": int(self.cortex.n),
            "edges": int(self.cortex.weights.size),
            "steps": int(self.steps),
            "measured_graph": bool(self.graph.measured),
            "mode": self.readout.mode,
            "sim_steps": self.sim_steps,
            "expected_reward": float(self.dopamine.expected),
            "drift": self.cortex.weight_drift(),
        }
        # np.savez_compressed appends .npz to a path; handing it an open file
        # keeps the name the caller asked for
        with open(path, "wb") as fh:
            np.savez_compressed(fh,
                                weights=self.cortex.weights,
                                gains=self.cortex.gains,
                                meta=json.dumps(meta))
        return path

    def load(self, path) -> "Brain":
        """Load trained weights onto this brain. Refuses different wiring."""
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["fingerprint"] != self.fingerprint:
            raise ValueError(
                "this checkpoint was trained on a different connectome "
                f"({meta['fingerprint']} vs {self.fingerprint}). Weights from "
                "one wiring mean nothing on another.")
        self.cortex.weights[:] = z["weights"]
        self.cortex.gains[:] = z["gains"]
        self.dopamine.expected = float(meta.get("expected_reward", 0.0))
        return self

    # ------------------------------------------------------------------ info --
    def report(self) -> dict:
        d = self.cortex.weight_drift()
        return {
            "neurons": int(self.cortex.n),
            "edges": d["edges"],
            "decisions": self.steps,
            "ms_per_decision": self.window_ms,
            "dopamine": round(self.dopamine.level, 4),
            "expects": round(self.dopamine.expected, 4),
            "synapses_changed": d["changed"],
            "mean_change_mv": round(d["mean_abs_change_mv"], 6),
            "reward_total": round(self.trace.total, 3),
            "reward_recent": round(self.trace.recent, 4),
            "success_rate": round(self.trace.success_rate, 3),
        }

    def __str__(self) -> str:
        r = self.report()
        return (f"Brain({r['neurons']:,} neurons, {r['edges']:,} synapses, "
                f"{r['decisions']} decisions, {r['synapses_changed']:,} changed, "
                f"{self.dopamine})")
