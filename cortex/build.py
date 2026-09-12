"""
Turn the public MICrONS release into a Graph.

Nothing here is downloaded for you. The files are public, need no account and
no key, and the exact list is in `SOURCES` below and in the README. Put them
under data/ and run:

    python -m cortex.build

Output: build/graph.npz, which Graph.load() reads.

Sign convention. The fly connectome gets it from predicted neurotransmitter.
Cortex ships no neurotransmitter call, so the sign comes from the cell-type
classifier: excitatory_neuron +1 (glutamate), inhibitory_neuron -1 (GABA).
Non-neurons - astrocyte, oligodendrocyte, microglia, OPC, pericyte - are
dropped, not zeroed.

If your counts differ from the ones this prints, one of us has a bug.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .graph import Graph
from .params import MIN_SYNAPSES, MV_PER_SYNAPSE

DATA = Path("data")
BUILD = Path("build")

SOURCES = """
storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1412/
  connections_with_nuclei.csv.gz                3.0 GB
  aibs_metamodel_celltypes_v661_merged.csv.gz   3.5 MB
  functional_properties_v3_bcm_merged.csv.gz    746 KB
  coregistration_manual_v4_merged.csv.gz        942 KB
bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/
  digital_twin_properties/v2/readout/readout_locations.npy   -> data/dt/readout_readout_locations.npy
  digital_twin_properties/v2/readout/units.csv               -> data/dt/readout_units.csv
  digital_twin_properties/v2/ori_dir_tuning/units.csv        -> data/dt/ori_dir_tuning_units.csv
  digital_twin_properties/v2/anatomy/units.csv               -> data/dt/anatomy_units.csv
  digital_twin_properties/v2/performance/units.csv           -> data/dt/performance_units.csv
"""

CELLTYPE_COLS = ("id target_id classification_system cell_type volume "
                 "pt_supervoxel_id pt_root_id x y z").split()
FUNC_COLS = ("id target_id session scan_idx unit_id pref_ori pref_dir gOSI gDSI "
             "cc_abs volume pt_supervoxel_id pt_root_id x y z").split()
CONN_COLS = "pre_pt_root_id post_pt_root_id n_syn sum_size pre_nuc_id post_nuc_id".split()
COREG_COLS = ("id target_id session scan_idx unit_id field residual score volume "
              "pt_supervoxel_id pt_root_id x y z").split()
KEY = ["session", "scan_idx", "unit_id"]


def twin_table(data: Path):
    """Join the digital-twin units to EM cells. That link is the whole design:
    it is what gives every cell a measured receptive-field centre."""
    import pandas as pd

    dt = data / "dt"
    ro = pd.read_csv(dt / "readout_units.csv").reset_index(drop=True)
    loc = np.load(dt / "readout_readout_locations.npy")
    ro["rf_x"], ro["rf_y"] = loc[:, 0], loc[:, 1]

    u = (ro.merge(pd.read_csv(dt / "ori_dir_tuning_units.csv"), on=KEY)
           .merge(pd.read_csv(dt / "anatomy_units.csv"), on=KEY)
           .merge(pd.read_csv(dt / "performance_units.csv")[KEY + ["cc_abs", "cc_norm"]],
                  on=KEY))
    cg = pd.read_csv(data / "coregistration_manual_v4_merged.csv.gz", names=COREG_COLS)
    m = u.merge(cg[KEY + ["pt_root_id", "residual", "score"]], on=KEY, how="inner")
    print(f"  twin units {len(ro):,} -> joined {len(u):,} -> "
          f"matched to an EM cell {len(m):,} ({m.pt_root_id.nunique():,} cells)")
    # one row per EM cell: keep the twin unit that predicts it best
    return m.sort_values("cc_abs", ascending=False).drop_duplicates("pt_root_id")


def main(data: Path = DATA, out: Path = BUILD / "graph.npz") -> Path:
    import pandas as pd

    data = Path(data)
    if not (data / "connections_with_nuclei.csv.gz").exists():
        raise FileNotFoundError(
            f"expected the MICrONS files under {data.resolve()}.\n{SOURCES}")

    print("cell types ...")
    t = pd.read_csv(data / "aibs_metamodel_celltypes_v661_merged.csv.gz",
                    names=CELLTYPE_COLS).drop_duplicates("pt_root_id")
    t = t[t.classification_system.isin(("excitatory_neuron", "inhibitory_neuron"))]
    t = t[t.pt_root_id > 0]
    exc = int((t.classification_system == "excitatory_neuron").sum())
    print(f"  {len(t):,} classified neurons ({exc:,} exc, {len(t) - exc:,} inh)")

    roots = np.sort(t.pt_root_id.unique())
    n = len(roots)
    idx = pd.Series(np.arange(n, dtype=np.int32), index=roots)
    ti = t.set_index("pt_root_id")

    sign = np.where(ti["classification_system"].reindex(roots).to_numpy()
                    == "excitatory_neuron", 1.0, -1.0).astype(np.float32)
    types = ti["cell_type"].reindex(roots).fillna("").to_numpy().astype(str)
    pos = np.stack([ti[c].reindex(roots).to_numpy() for c in "xyz"], 1).astype(np.float32)

    print(f"connections (keeping pairs with >= {MIN_SYNAPSES} synapses) ...")
    keep_r, keep_c, keep_v, n_rows = [], [], [], 0
    valid = pd.Index(roots)
    reader = pd.read_csv(data / "connections_with_nuclei.csv.gz", names=CONN_COLS,
                         usecols=[0, 1, 2], chunksize=4_000_000)
    for k, ch in enumerate(reader):
        n_rows += len(ch)
        ch = ch[ch.n_syn >= MIN_SYNAPSES]
        ch = ch[ch.pre_pt_root_id.isin(valid) & ch.post_pt_root_id.isin(valid)]
        if len(ch):
            keep_r.append(idx.loc[ch.post_pt_root_id].to_numpy())   # row = post
            keep_c.append(idx.loc[ch.pre_pt_root_id].to_numpy())    # col = pre
            keep_v.append(ch.n_syn.to_numpy().astype(np.float32))
        print(f"  chunk {k:>3}: {n_rows:>12,} rows, "
              f"{sum(len(a) for a in keep_v):>10,} edges kept", end="\r")
    print()

    r = np.concatenate(keep_r)
    c = np.concatenate(keep_c)
    w = np.concatenate(keep_v)
    W = sp.csr_matrix((w * MV_PER_SYNAPSE * sign[c], (r, c)), shape=(n, n),
                      dtype=np.float32)
    W.sum_duplicates()
    print(f"  {n_rows:,} pairs on disk -> {W.nnz:,} edges, {int(w.sum()):,} synapses")
    print(f"  excitatory {(W.data > 0).sum():,}   inhibitory {(W.data < 0).sum():,}")

    print("direction tuning ...")
    f = pd.read_csv(data / "functional_properties_v3_bcm_merged.csv.gz", names=FUNC_COLS)
    f = f.sort_values("cc_abs", ascending=False).drop_duplicates("pt_root_id")
    fi = f[f.pt_root_id.isin(valid)].set_index("pt_root_id")
    pref_dir = fi["pref_dir"].reindex(roots).to_numpy().astype(np.float32)
    gDSI = fi["gDSI"].reindex(roots).to_numpy().astype(np.float32)
    cc_abs = fi["cc_abs"].reindex(roots).to_numpy().astype(np.float32)
    print(f"  {np.isfinite(pref_dir).sum():,} cells with measured tuning")

    print("receptive fields ...")
    rf = np.full((n, 2), np.nan, dtype=np.float32)
    try:
        tw = twin_table(data).set_index("pt_root_id")
        here = tw.index.intersection(roots)
        rows = idx.loc[here].to_numpy()
        rf[rows, 0] = tw.loc[here, "rf_x"].to_numpy()
        rf[rows, 1] = tw.loc[here, "rf_y"].to_numpy()
        # the twin's own numbers are better than the merged table where both exist
        for col, arr in (("pref_dir", pref_dir), ("gDSI", gDSI), ("cc_abs", cc_abs)):
            arr[rows] = tw.loc[here, col].to_numpy().astype(np.float32)
        print(f"  {len(here):,} cells with a measured receptive-field centre")
    except FileNotFoundError as e:
        print(f"  no digital-twin files ({e.filename}); the graph will have no "
              f"receptive fields and Retina will refuse to build")

    g = Graph(W=W.tocsc(), sign=sign, types=types, pos=pos, pref_dir=pref_dir,
              gDSI=gDSI, cc_abs=cc_abs, rf=rf, roots=roots, measured=True)
    path = g.save(out)
    print(f"\n{g.describe()}")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


if __name__ == "__main__":
    main()
