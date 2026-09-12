"""
Fix the excitation/inhibition skew that comes from how the volume was proofread.

In the measured cube, inhibitory cells have a mean out-degree of 212 and
excitatory cells 10.4. That is not cortex, that is reconstruction: interneuron
axons are local and stay inside the cubic millimetre, pyramidal axons leave it
and get cut. Left alone the network sits at -74 mV, below its own resting
potential, because the only complete axons in the data are the ones that inhibit.

So outgoing weights get one scalar per cell, in two corrections:

  1. inhibition is scaled down until the total weight hits the textbook
     cortical split, 80% excitatory
  2. a cell whose axon was cut has too few surviving outputs, so its remaining
     synapses under-represent it. Each cell is normalised toward the median of
     its own type, capped, so a cell with two surviving synapses does not
     become a hub.

Both are modelling choices made by a person. The wiring is measured; this is not.
"""
import numpy as np
import scipy.sparse as sp

from .params import EI_TARGET


def per_neuron_gains(graph, ei_target: float = EI_TARGET, verbose: bool = True):
    """Return an (n,) gain vector multiplying each neuron's outgoing weights."""
    W = graph.W.tocsc()                       # column j = the outputs of neuron j
    absW = sp.csc_matrix((np.abs(W.data), W.indices, W.indptr), shape=W.shape)
    out_w = np.asarray(absW.sum(axis=0)).ravel()

    exc, inh = graph.sign > 0, graph.sign < 0
    e_tot, i_tot = out_w[exc].sum(), out_w[inh].sum()
    g = np.ones(graph.n, dtype=np.float32)
    if i_tot > 0:
        g[inh] = np.float32(e_tot / (ei_target * i_tot))
    if verbose:
        ratio = e_tot / i_tot if i_tot else float("inf")
        print(f"raw totals   exc |w| {e_tot:,.0f}   inh |w| {i_tot:,.0f}   "
              f"E/I {ratio:.2f}  (target {ei_target})")
        print(f"inhibitory gain {g[inh][0] if inh.any() else 1.0:.4f} "
              f"(excitatory left at 1.0)")

    for t in np.unique(graph.types):
        m = graph.types == t
        w = out_w[m]
        live = w > 0
        if live.sum() < 20:
            continue
        med = np.median(w[live])
        adj = np.ones(int(m.sum()), dtype=np.float32)
        adj[live] = np.clip(med / w[live], 0.25, 4.0)
        g[m] *= adj

    if verbose:
        print(f"gains: min {g.min():.3f}  median {np.median(g):.3f}  max {g.max():.3f}")
    return g
