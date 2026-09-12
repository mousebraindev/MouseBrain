"""
The socket: what you implement to attach the brain to your own problem.

A Task is four methods. Everything the repo ships - the target hunt, the
bandit, the browser - is written against this and nothing else, so anything you
write against it gets the same training loop, the same reward machinery and the
same checkpoints.

    class MyTask(Task):
        def reset(self): ...
        def observe(self) -> np.ndarray:     # a 2D field, any scale
        def act(self, action) -> float:      # do it, return the reward
        def done(self) -> bool: ...

The contract in one line: you render the world into a picture, the brain points
somewhere in it, you say whether that was good.

Reward, stated plainly, because this is where people fool themselves: return
something you would still be happy to report if the brain learned to maximise
it by cheating. Reward every step, not only at the end - the eligibility traces
reach back about a second, not across an episode.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np

from .readout import Action


class Task(ABC):
    """Anything a brain can be attached to."""

    name: str = "task"
    #: how many options, if this task wants a discrete choice instead of a vector
    choices: Optional[int] = None

    @abstractmethod
    def reset(self) -> None:
        """Start a fresh episode."""

    @abstractmethod
    def observe(self) -> np.ndarray:
        """The world as a 2D field. Values in 0..1; anything larger is scaled."""

    @abstractmethod
    def act(self, action: Action) -> float:
        """Carry out the decision and return the reward it earned."""

    def done(self) -> bool:
        """True when the episode is over. Default: never, so max_steps ends it."""
        return False

    def succeeded(self) -> bool:
        """True if this episode counts as solved. Used for reporting only."""
        return False

    # optional, for tasks with a cursor or an attention point
    def focus(self) -> Optional[Tuple[float, float]]:
        return None

    def window(self) -> Optional[Tuple[float, float]]:
        return None

    def view(self) -> Optional[Tuple[float, float]]:
        """Caller units spanned by one gaze. Defaults to the window."""
        return None


# --------------------------------------------------------------------- loop --
class Episode:
    """What one attempt did."""

    def __init__(self, index: int):
        self.index = index
        self.reward = 0.0
        self.steps = 0
        self.success = False
        self.hz: List[float] = []
        self.dopamine: List[float] = []
        self.seconds = 0.0

    def as_dict(self) -> dict:
        return {"episode": self.index, "reward": round(self.reward, 4),
                "steps": self.steps, "success": self.success,
                "median_hz": round(float(np.median(self.hz)) if self.hz else 0.0, 2),
                "peak_dopamine": round(max(self.dopamine) if self.dopamine else 0.0, 3),
                "seconds": round(self.seconds, 2)}


def run_episode(brain, task: Task, max_steps: int = 60,
                learn: bool = True,
                on_step: Optional[Callable[[int, Action, float], None]] = None) -> Episode:
    """One attempt: look, act, be rewarded, repeat."""
    ep = Episode(0)
    t0 = time.time()
    task.reset()
    brain.new_episode()

    for t in range(max_steps):
        field = task.observe()
        act = brain.look(field,
                         focus=task.focus(),
                         window=task.window(),
                         view=task.view(),
                         k=task.choices)
        r = float(task.act(act))

        if learn:
            brain.reward(r, success=task.succeeded())
        else:
            brain.trace.add(r, task.succeeded())

        ep.reward += r
        ep.steps = t + 1
        ep.hz.append(act.hz)
        ep.dopamine.append(brain.dopamine.level)
        if on_step is not None:
            on_step(t, act, r)
        if task.done():
            break

    ep.success = task.succeeded()
    ep.seconds = time.time() - t0
    return ep


def train(brain, task: Task, episodes: int = 20, max_steps: int = 60,
          verbose: bool = True,
          on_episode: Optional[Callable[[Episode], None]] = None) -> List[Episode]:
    """
    Run the task over and over, paying the brain as it goes.

    No gradients, no targets, no replay buffer. The only thing crossing from
    the task into the brain is a scalar per step.
    """
    out: List[Episode] = []
    for i in range(episodes):
        ep = run_episode(brain, task, max_steps=max_steps, learn=True)
        ep.index = i
        out.append(ep)
        if on_episode is not None:
            on_episode(ep)
        if verbose:
            drift = brain.cortex.weight_drift()
            print(f"  ep {i:>3}  reward {ep.reward:+8.2f}  steps {ep.steps:>3}  "
                  f"{'solved' if ep.success else '      '}  "
                  f"da {max(ep.dopamine) if ep.dopamine else 0:+.2f}  "
                  f"synapses moved {drift['changed']:,}")
    return out


def evaluate(brain, task: Task, episodes: int = 10, max_steps: int = 60,
             verbose: bool = True) -> dict:
    """
    Measure the brain without letting it learn from the measurement.

    Plasticity is frozen for the duration, so the number that comes out is what
    the brain currently knows and not what it picked up while being tested.
    """
    was = brain.plasticity
    brain.freeze()
    rewards, wins, steps = [], 0, []
    try:
        for _ in range(episodes):
            ep = run_episode(brain, task, max_steps=max_steps, learn=False)
            rewards.append(ep.reward)
            steps.append(ep.steps)
            wins += int(ep.success)
    finally:
        brain.plasticity = was

    out = {"episodes": episodes,
           "mean_reward": float(np.mean(rewards)),
           "median_reward": float(np.median(rewards)),
           "success_rate": wins / max(episodes, 1),
           "mean_steps": float(np.mean(steps))}
    if verbose:
        print(f"  {episodes} episodes: mean reward {out['mean_reward']:+.2f}, "
              f"solved {out['success_rate'] * 100:.0f}%, "
              f"{out['mean_steps']:.1f} steps")
    return out
