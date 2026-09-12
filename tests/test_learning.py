import numpy as np
import pytest

from cortex import Brain, Cortex, Dopamine, Graph, RewardModulatedSTDP


@pytest.fixture
def wired():
    g = Graph.synthetic(n=300, seed=1)
    cx = Cortex(g)
    return g, cx, RewardModulatedSTDP(cx)


def _drive(cx, stdp, steps=300, rate=80.0, seed=0):
    idx = np.arange(0, cx.n, 2)
    return cx.run({tuple(idx): rate}, steps=steps, plasticity=stdp, seed=seed)


# ----------------------------------------------------------------- the rules --
def test_no_dopamine_no_change(wired):
    """Eligibility on its own moves nothing. That is the whole point of the
    third factor: activity is not learning."""
    _, cx, stdp = wired
    before = cx.weights.copy()
    _drive(cx, stdp)
    assert np.array_equal(cx.weights, before)
    assert stdp.stats()["eligible_edges"] > 0     # but something was eligible


def test_dopamine_moves_weights(wired):
    _, cx, stdp = wired
    _drive(cx, stdp)
    moved = stdp.apply(1.0)
    assert moved > 0
    assert not np.array_equal(cx.weights, cx.base_weights)


def test_dale_law_holds_under_learning(wired):
    """An excitatory cell cannot learn its way into being inhibitory."""
    g, cx, stdp = wired
    for r in (+3.0, -3.0, +3.0, -3.0):
        _drive(cx, stdp, seed=int(abs(r) * 10))
        stdp.apply(r)
    live = cx.base_weights != 0
    assert np.all(np.sign(cx.weights[live]) * np.sign(cx.base_weights[live]) >= 0)


def test_ceiling_holds(wired):
    """No synapse grows past its measured strength times the cap."""
    _, cx, stdp = wired
    for _ in range(6):
        _drive(cx, stdp)
        stdp.apply(4.0)
    assert np.all(np.abs(cx.weights) <= np.abs(cx.base_weights) * 3.0 + 1e-5)


def test_punishment_is_the_opposite_of_reward(wired):
    """Same eligibility, opposite sign of dopamine, opposite weight change.

    Compared edge by edge, not in total: synaptic scaling holds the total where
    the microscope found it, so the sum barely moves in either direction. What
    must hold is that the two payouts push each synapse opposite ways.
    """
    _, cx, stdp = wired
    _drive(cx, stdp)
    stdp.renormalise()
    e = stdp.e.copy()

    stdp.apply(+1.0)
    on_reward = cx.weights - cx.base_weights

    cx.reset_weights()
    stdp.e[:] = e
    stdp.apply(-1.0)
    on_punishment = cx.weights - cx.base_weights

    moved = (np.abs(on_reward) > 1e-9) | (np.abs(on_punishment) > 1e-9)
    assert moved.sum() > 100
    r = np.corrcoef(on_reward[moved], on_punishment[moved])[0, 1]
    assert r < -0.5, f"reward and punishment should oppose, correlation was {r:.3f}"


def test_homeostasis_holds_total_input_weight(wired):
    """Learning is a competition between a cell's inputs, not a volume knob."""
    _, cx, stdp = wired
    for _ in range(4):
        _drive(cx, stdp)
        stdp.apply(3.0)
    post = cx.indices
    before = np.bincount(post, weights=np.abs(cx.base_weights), minlength=cx.n)
    after = np.bincount(post, weights=np.abs(cx.weights), minlength=cx.n)
    live = before > 0
    assert np.allclose(after[live], before[live], rtol=0.25)


def test_clearing_traces_stops_the_payout(wired):
    """Eligibility from a previous episode must not be cashed by this one."""
    _, cx, stdp = wired
    _drive(cx, stdp)
    stdp.clear_traces()
    assert stdp.apply(2.0) == 0.0
    assert np.array_equal(cx.weights, cx.base_weights)


def test_eligibility_decays(wired):
    _, cx, stdp = wired
    _drive(cx, stdp, steps=200)
    stdp.renormalise()
    hot = np.abs(stdp.e).sum()
    cx.run(None, steps=4000, plasticity=stdp)      # 800 ms of silence
    stdp.renormalise()
    assert np.abs(stdp.e).sum() < hot * 0.5


# ------------------------------------------------------------------ dopamine --
def test_expected_reward_stops_teaching():
    """A reward you always get is not a surprise, and a dopamine cell knows it."""
    da = Dopamine(rpe_tau=10)
    first = da.deliver(1.0)
    for _ in range(200):
        last = da.deliver(1.0)
    assert first == pytest.approx(1.0)
    assert abs(last) < 0.05


def test_dopamine_decays_toward_tonic():
    da = Dopamine(tonic=0.0, tau_ms=100.0)
    da.deliver(2.0)
    da.decay(500.0)
    assert abs(da.level) < 0.05


def test_arousal_changes_behaviour_not_only_synapses():
    """Reward changes what the animal does now, before any weight has moved."""
    da = Dopamine()
    quiet = (da.explore_sigma(1.0), da.approach_gain(1.0))
    da.deliver(3.0)
    excited = (da.explore_sigma(1.0), da.approach_gain(1.0))
    assert excited[0] < quiet[0]        # less casting about
    assert excited[1] > quiet[1]        # more commitment


# --------------------------------------------------------------------- brain --
def test_frozen_brain_does_not_learn():
    b = Brain.demo(n=300, sim_steps=60, quiet=True).freeze()
    f = np.zeros((40, 40), np.float32)
    f[10:20, 10:20] = 1.0
    b.look(f)
    b.reward(5.0)
    assert b.cortex.weight_drift()["changed"] == 0


def test_checkpoint_round_trip(tmp_path):
    a = Brain.demo(n=300, sim_steps=60, quiet=True, seed=2)
    f = np.zeros((40, 40), np.float32)
    f[5:15, 20:30] = 1.0
    a.look(f)
    a.reward(1.0)
    p = a.save(tmp_path / "run.brain")

    b = Brain.demo(n=300, sim_steps=60, quiet=True, seed=2)
    b.load(p)
    assert np.allclose(a.cortex.weights, b.cortex.weights)


def test_checkpoint_refuses_different_wiring(tmp_path):
    a = Brain.demo(n=300, sim_steps=60, quiet=True, seed=2)
    p = a.save(tmp_path / "a.brain")
    other = Brain.demo(n=320, sim_steps=60, quiet=True, seed=5)
    with pytest.raises(ValueError, match="different connectome"):
        other.load(p)
