import numpy as np
import pytest

from cortex import Cortex, Graph


@pytest.fixture(scope="module")
def graph():
    return Graph.synthetic(n=400, seed=0)


def test_synthetic_graph_is_labelled_synthetic(graph):
    assert graph.measured is False
    assert "SYNTHETIC" in graph.describe()


def test_signs_follow_the_cell_not_the_edge(graph):
    """Dale's law in the wiring: every edge out of a cell has that cell's sign."""
    W = graph.W.tocsc()
    for j in range(0, graph.n, 37):
        col = W.data[W.indptr[j]:W.indptr[j + 1]]
        if col.size:
            assert np.all(np.sign(col) == graph.sign[j])


def test_it_spikes(graph):
    cx = Cortex(graph)
    idx = np.arange(0, graph.n, 3)
    out = cx.run({tuple(idx): 60.0}, steps=200, record={"all": idx})
    assert out["_spikes_per_sec"] > 0
    assert len(out["_fired"]) > 0


def test_silence_in_silence_out(graph):
    cx = Cortex(graph)
    out = cx.run(None, steps=200)
    assert out["_spikes_per_sec"] == 0.0


def test_same_seed_same_run(graph):
    cx = Cortex(graph)
    idx = np.arange(0, graph.n, 3)
    a = cx.run({tuple(idx): 40.0}, steps=120, seed=7)
    b = cx.run({tuple(idx): 40.0}, steps=120, seed=7)
    assert a["_spikes_per_sec"] == b["_spikes_per_sec"]


def test_state_carries_between_calls(graph):
    """A brain that reboots between frames has no memory of what it was doing."""
    cx = Cortex(graph)
    idx = np.arange(0, graph.n, 3)
    first = cx.run({tuple(idx): 60.0}, steps=200, seed=1)
    carried = cx.run({tuple(idx): 60.0}, steps=200, seed=2, state=first["_state"])
    fresh = cx.run({tuple(idx): 60.0}, steps=200, seed=2)
    assert carried["_mean_mv"] != fresh["_mean_mv"]


def test_run_does_not_mutate_the_state_it_was_given(graph):
    cx = Cortex(graph)
    idx = np.arange(0, graph.n, 3)
    st = cx.fresh_state()
    before = st.v.copy()
    cx.run({tuple(idx): 60.0}, steps=100, state=st)
    assert np.array_equal(st.v, before)


def test_out_slots_match_scipy(graph):
    cx = Cortex(graph)
    fired = np.array([1, 5, 9, 40])
    slots = cx.out_slots(fired)
    assert np.array_equal(cx.edge_pre[slots], np.repeat(
        fired, np.diff(cx.indptr)[fired]).astype(np.int32))
