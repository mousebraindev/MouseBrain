"""
mousecortex - a cubic millimetre of mouse visual cortex you can attach to
anything, and train with reward.

    from cortex import Brain
    from cortex.tasks import ReachTarget

    brain = Brain.demo()
    brain.look(task.observe())
    brain.reward(+1.0)

The wiring is measured. The learning rule, the reward, and every constant
marked CHOSEN in params.py are modelling decisions made by people.
"""
from .brain import Brain
from .core import Cortex, State
from .field import Retina, render_points
from .graph import Graph
from .params import Passive
from .plasticity import Frozen, RewardModulatedSTDP
from .readout import Action, Readout
from .reward import Dopamine, RewardTrace
from .task import Task, evaluate, run_episode, train

__version__ = "0.1.0"
__all__ = [
    "Brain", "Cortex", "State", "Graph", "Retina", "Readout", "Action",
    "Dopamine", "RewardTrace", "RewardModulatedSTDP", "Frozen", "Passive",
    "Task", "train", "evaluate", "run_episode", "render_points",
]
