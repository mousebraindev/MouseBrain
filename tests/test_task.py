import numpy as np
import pytest

from cortex import Brain, Task, evaluate, render_points, run_episode, train
from cortex.tasks import CueChoice, ReachTarget


def test_render_points_is_a_field():
    f = render_points([[10, 10], [40, 30]], size=(64, 48), radius=4)
    assert f.shape == (48, 64)
    assert 0.0 <= f.min() and f.max() == pytest.approx(1.0)


def test_reach_target_runs():
    b = Brain.demo(n=400, sim_steps=60, quiet=True)
    ep = run_episode(b, ReachTarget(size=(80, 60), seed=0), max_steps=6)
    assert ep.steps > 0
    assert np.isfinite(ep.reward)


def test_cue_choice_answers_in_range():
    task = CueChoice(k=4, size=(64, 64), seed=0)
    b = Brain.demo(n=600, sim_steps=120, choices=4, quiet=True)
    task.reset()
    act = b.look(task.observe(), k=4)
    assert -1 <= act.choice < 4


def test_choice_without_a_pool_is_refused():
    b = Brain.demo(n=300, sim_steps=60, quiet=True)
    with pytest.raises(ValueError, match="motor pool"):
        b.look(np.zeros((40, 40), np.float32), k=4)


def test_evaluate_does_not_teach():
    """Measuring a brain must not change it, or the measurement is worthless."""
    b = Brain.demo(n=400, sim_steps=60, quiet=True)
    task = ReachTarget(size=(80, 60), seed=1)
    train(b, task, episodes=1, max_steps=3, verbose=False)
    before = b.cortex.weights.copy()
    evaluate(b, task, episodes=2, max_steps=3, verbose=False)
    assert np.array_equal(b.cortex.weights, before)
    assert b.plasticity.enabled, "evaluate must hand plasticity back"


def test_a_custom_task_needs_four_methods():
    """The socket, as advertised."""

    class Blink(Task):
        name = "blink"

        def __init__(self):
            self.t = 0

        def reset(self):
            self.t = 0

        def observe(self):
            f = np.zeros((48, 48), np.float32)
            f[20:28, 20:28] = 1.0 if self.t % 2 else 0.2
            return f

        def act(self, action):
            self.t += 1
            return 1.0 if action.dx > 0 else -1.0

        def done(self):
            return self.t >= 4

    b = Brain.demo(n=400, sim_steps=60, quiet=True)
    ep = run_episode(b, Blink(), max_steps=8)
    assert ep.steps == 4
