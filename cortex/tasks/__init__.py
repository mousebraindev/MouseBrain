"""Tasks that ship with the repo. ScreenTask is imported lazily: it needs a
browser, everything else needs numpy."""
from .cue import CueChoice
from .target import ReachTarget

__all__ = ["ReachTarget", "CueChoice", "ScreenTask"]


def __getattr__(name):
    if name == "ScreenTask":
        from .screen import ScreenTask
        return ScreenTask
    raise AttributeError(name)
