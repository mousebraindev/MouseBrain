"""
Attach the brain to something of your own.

A Task is four methods. Here is one that is not a picture at all: a list of
numbers, drawn as a field, where the brain is paid for pointing at the largest.
Anything you can render, the cortex can look at.

    python examples/04_your_own_task.py
"""
import numpy as np

from cortex import Brain, Task, render_points, train


class PickTheLargest(Task):
    """Six bars. Point at the tallest."""

    name = "pick-the-largest"

    def __init__(self, bars=6, size=(180, 120), seed=0):
        self.bars, self.w, self.h = bars, size[0], size[1]
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.values = self.rng.random(self.bars)
        self.cursor = np.array([self.w / 2, self.h / 2], dtype=np.float32)
        self.spent = 0

    def _centres(self):
        xs = np.linspace(20, self.w - 20, self.bars)
        ys = self.h - 10 - self.values * (self.h - 40)
        return np.column_stack([xs, ys])

    def observe(self):
        # the value is the brightness, so a visual area can see the difference
        return render_points(self._centres(), size=(self.w, self.h),
                             radius=9.0, weights=self.values)

    def act(self, action):
        self.spent += 1
        self.cursor += np.array([action.dx, action.dy], dtype=np.float32)
        self.cursor[0] = np.clip(self.cursor[0], 0, self.w - 1)
        self.cursor[1] = np.clip(self.cursor[1], 0, self.h - 1)

        target = self._centres()[int(np.argmax(self.values))]
        d = float(np.linalg.norm(self.cursor - target))
        # paid every step for being close, because eligibility traces reach
        # back about a second, not across an episode
        return float(np.exp(-d / 40.0)) - 0.05

    def done(self):
        return self.spent >= 12

    def succeeded(self):
        target = self._centres()[int(np.argmax(self.values))]
        return float(np.linalg.norm(self.cursor - target)) < 15.0


if __name__ == "__main__":
    brain = Brain.demo(n=2000, sim_steps=200, explore=1.5)
    train(brain, PickTheLargest(), episodes=25, max_steps=12)
    print("\n" + str(brain))
