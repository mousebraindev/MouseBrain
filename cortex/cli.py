"""
Command line: cortex demo | train | eval | build | info

    cortex demo                     30 seconds, synthetic wiring, no download
    cortex train --episodes 40      train on the reach task, save a checkpoint
    cortex eval --load run.brain    measure a saved brain, learning frozen
    cortex build                    build the measured connectome from data/
    cortex info                     what is loaded and how big it is
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _brain(args, **kw):
    from .brain import Brain
    if args.measured:
        return Brain.measured(sim_steps=args.sim_steps, **kw)
    return Brain.demo(n=args.neurons, seed=args.seed, sim_steps=args.sim_steps, **kw)


def _task(args):
    from .tasks import CueChoice, ReachTarget
    if args.task == "cue":
        return CueChoice(k=4, seed=args.seed)
    return ReachTarget(seed=args.seed)


def cmd_demo(args) -> int:
    from .task import evaluate, train
    task = _task(args)
    brain = _brain(args, explore=2.0, choices=task.choices)
    print(f"\ntask: {task.name}\n")

    print("before training, plasticity frozen:")
    before = evaluate(brain, task, episodes=args.eval, max_steps=args.max_steps)

    print("\ntraining:")
    train(brain, task, episodes=args.episodes, max_steps=args.max_steps)

    print("\nafter training, plasticity frozen:")
    after = evaluate(brain, task, episodes=args.eval, max_steps=args.max_steps)

    d = brain.cortex.weight_drift()
    print(f"\nmean reward {before['mean_reward']:+.2f} -> {after['mean_reward']:+.2f}"
          f"   solved {before['success_rate']*100:.0f}% -> {after['success_rate']*100:.0f}%")
    print(f"synapses moved: {d['changed']:,} of {d['edges']:,} "
          f"(mean {d['mean_abs_change_mv']:.5f} mV)")
    print("\nOne run of a stochastic system on synthetic wiring. If you want a "
          "number you can quote, run it over several seeds.")
    return 0


def cmd_train(args) -> int:
    from .task import evaluate, train
    task = _task(args)
    brain = _brain(args, explore=2.0, choices=task.choices)
    if args.load:
        brain.load(args.load)
        print(f"loaded {args.load}")
    train(brain, task, episodes=args.episodes, max_steps=args.max_steps)
    evaluate(brain, task, episodes=args.eval, max_steps=args.max_steps)
    out = brain.save(args.save)
    print(f"wrote {out}")
    return 0


def cmd_eval(args) -> int:
    from .task import evaluate
    task = _task(args)
    brain = _brain(args, choices=task.choices)
    if args.load:
        brain.load(args.load)
        print(f"loaded {args.load}")
    evaluate(brain, task, episodes=args.eval, max_steps=args.max_steps)
    return 0


def cmd_build(args) -> int:
    from .build import main as build_main
    build_main()
    return 0


def cmd_info(args) -> int:
    from .graph import Graph
    try:
        g = Graph.load()
        print("measured graph:", g.describe())
    except FileNotFoundError as e:
        print(f"no measured graph: {e}".splitlines()[0])
    g = Graph.synthetic(n=args.neurons, seed=args.seed)
    print("synthetic graph:", g.describe())
    return 0


def _common() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--measured", action="store_true",
                   help="use the MICrONS connectome from build/graph.npz")
    c.add_argument("--neurons", type=int, default=3000,
                   help="synthetic population size")
    c.add_argument("--sim-steps", type=int, default=200, dest="sim_steps",
                   help="dt steps per decision (200 = 40 ms of brain time)")
    c.add_argument("--task", choices=("reach", "cue"), default="reach")
    c.add_argument("--episodes", type=int, default=30)
    c.add_argument("--eval", type=int, default=10)
    c.add_argument("--max-steps", type=int, default=40, dest="max_steps")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--save", default="checkpoints/run.brain")
    c.add_argument("--load", default=None)
    return c


def main(argv=None) -> int:
    common = _common()
    p = argparse.ArgumentParser(prog="cortex", description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    for name, fn in (("demo", cmd_demo), ("train", cmd_train), ("eval", cmd_eval),
                     ("build", cmd_build), ("info", cmd_info)):
        sp = sub.add_parser(name, parents=[common],
                            help=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else None)
        sp.set_defaults(fn=fn)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 1
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
