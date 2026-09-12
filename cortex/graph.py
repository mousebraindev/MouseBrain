"""
The wiring: who connects to whom, how strongly, and with what sign.

A Graph is everything the microscope measured, plus nothing. It carries no
learning, no state and no task. Two ways to get one:

    Graph.load()        the measured MICrONS cube, after `cortex build`
    Graph.synthetic()   a statistically matched stand-in that needs no download

The synthetic graph exists so the repo runs, tests and trains on a laptop with
no network. It is generated, not measured, and it says so in `.measured`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import params

DEFAULT_PATH = Path("build") / "graph.npz"


@dataclass
class Graph:
    """A signed sparse connectome. Row = postsynaptic, column = presynaptic."""

    W: sp.csc_matrix          # mV delivered to the post cell per presynaptic spike
    sign: np.ndarray          # (n,) +1 excitatory, -1 inhibitory
    types: np.ndarray         # (n,) cell type label, e.g. "23P", "BC"
    pos: np.ndarray           # (n, 3) soma position in the volume, micrometres
    pref_dir: np.ndarray      # (n,) preferred direction of motion, degrees, or NaN
    gDSI: np.ndarray          # (n,) how direction-selective the cell is, 0..1
    cc_abs: np.ndarray        # (n,) how well the digital twin predicts this cell
    rf: np.ndarray            # (n, 2) receptive-field centre in stimulus space, or NaN
    roots: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    measured: bool = True     # False for a synthetic stand-in

    # ------------------------------------------------------------------ info --
    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def n_edges(self) -> int:
        return self.W.nnz

    def describe(self) -> str:
        e = int((self.sign > 0).sum())
        i = int((self.sign < 0).sum())
        tuned = int(np.isfinite(self.pref_dir).sum())
        seen = int(np.isfinite(self.rf[:, 0]).sum())
        origin = "measured" if self.measured else "SYNTHETIC (generated, not measured)"
        return (f"{self.n:,} neurons ({e:,} exc / {i:,} inh), "
                f"{self.n_edges:,} edges, {tuned:,} with a direction, "
                f"{seen:,} with a receptive field - {origin}")

    # ------------------------------------------------------------------- io --
    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "Graph":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no connectome at {path}.\n"
                "Either run `cortex build` (needs the public MICrONS csv files, "
                "see README), or use Graph.synthetic() to work without it.")
        z = np.load(path, allow_pickle=False)
        W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"])).tocsc()
        return cls(
            W=W,
            sign=z["sign"],
            types=z["types"].astype(str),
            pos=z["pos"],
            pref_dir=z["pref_dir"],
            gDSI=z["gDSI"],
            cc_abs=z["cc_abs"],
            rf=z["rf"] if "rf" in z else np.full((W.shape[0], 2), np.nan, np.float32),
            roots=z["roots"] if "roots" in z else np.arange(W.shape[0]),
            measured=bool(z["measured"]) if "measured" in z else True,
        )

    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W = self.W.tocsr()
        np.savez_compressed(
            path,
            data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape,
            sign=self.sign, types=self.types, pos=self.pos,
            pref_dir=self.pref_dir, gDSI=self.gDSI, cc_abs=self.cc_abs,
            rf=self.rf, roots=self.roots, measured=self.measured,
        )
        return path

    # ------------------------------------------------------------ synthetic --
    @classmethod
    def synthetic(cls, n: int = 4000, seed: int = 0,
                  exc_fraction: float = 0.89) -> "Graph":
        """
        A cortex-shaped network with no cortex in it.

        The statistics are copied from the measured cube - 89% excitatory cells,
        log-normal synapse counts per pair, sparse connectivity, a retinotopic
        map with local scatter, direction preferences clustered rather than
        uniform - so code written against this runs unchanged against the real
        thing. The wiring itself is random. Nothing here is evidence about
        a mouse.
        """
        rng = np.random.default_rng(seed)

        sign = np.where(rng.random(n) < exc_fraction, 1.0, -1.0).astype(np.float32)
        types = np.where(sign > 0, "23P", "BC").astype(str)

        # a slab 1000 x 1000 um wide and 800 um deep, like the measured volume
        pos = np.column_stack([
            rng.uniform(0, 1000, n), rng.uniform(0, 800, n), rng.uniform(0, 1000, n),
        ]).astype(np.float32)

        # Retinotopy: depth-independent map of the visual field, plus scatter.
        rf = np.column_stack([
            pos[:, 0] / 1000.0 * 2 - 1 + rng.normal(0, 0.06, n),
            pos[:, 2] / 1000.0 * 2 - 1 + rng.normal(0, 0.06, n),
        ]).astype(np.float32)
        rf = np.clip(rf, -1, 1)

        # Direction preference is not uniform round the clock in the real data
        # either: a few cardinal clusters, plus a uniform floor.
        centres = np.array([0.0, 90.0, 180.0, 270.0])
        pick = rng.integers(0, len(centres), n)
        pref = (centres[pick] + rng.normal(0, 28, n)) % 360.0
        uniform = rng.random(n) < 0.35
        pref[uniform] = rng.uniform(0, 360, uniform.sum())
        gDSI = np.clip(rng.beta(1.6, 5.0, n), 0, 1).astype(np.float32)
        cc_abs = np.clip(rng.beta(3.0, 4.0, n), 0, 1).astype(np.float32)

        # Connectivity: inhibitory cells wire locally and densely, excitatory
        # cells sparsely and further - the same asymmetry the cube shows.
        deg_exc, deg_inh = 12, 90
        pre, post, nsyn = [], [], []
        for j in range(n):
            k = deg_inh if sign[j] < 0 else deg_exc
            k = max(1, int(rng.poisson(k)))
            d = np.linalg.norm(pos - pos[j], axis=1)
            scale = 120.0 if sign[j] < 0 else 260.0
            p = np.exp(-d / scale)
            p[j] = 0.0
            p /= p.sum()
            tgt = rng.choice(n, size=min(k, n - 1), replace=False, p=p)
            pre.append(np.full(len(tgt), j))
            post.append(tgt)
            nsyn.append(np.maximum(params.MIN_SYNAPSES,
                                   rng.lognormal(0.8, 0.7, len(tgt)).round()))
        pre = np.concatenate(pre)
        post = np.concatenate(post)
        nsyn = np.concatenate(nsyn).astype(np.float32)

        v = nsyn * params.MV_PER_SYNAPSE * sign[pre]
        W = sp.csr_matrix((v, (post, pre)), shape=(n, n), dtype=np.float32)
        W.sum_duplicates()

        return cls(W=W.tocsc(), sign=sign, types=types, pos=pos,
                   pref_dir=pref.astype(np.float32), gDSI=gDSI, cc_abs=cc_abs,
                   rf=rf, roots=np.arange(n, dtype=np.int64), measured=False)
