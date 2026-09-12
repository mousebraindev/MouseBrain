"""
Train a brain, measure it honestly, save it.

The measurement is the point. Plasticity is frozen during evaluation, so the
number that comes out is what the brain knows and not what it picked up while
being tested, and the comparison is against the same brain before training.

    python examples/03_train_on_a_task.py
"""
from cortex import Brain
from cortex.task import evaluate, train
from cortex.tasks import CueChoice

task = CueChoice(k=4, seed=7)
brain = Brain.demo(n=2000, sim_steps=250, choices=task.choices)

print(f"\ntask: {task.name}, {task.choices} options, chance {task.chance:.0%}\n")

before = evaluate(brain, task, episodes=120, max_steps=1)
train(brain, task, episodes=600, max_steps=1, verbose=False)
after = evaluate(brain, task, episodes=120, max_steps=1)

print(f"\nsolved {before['success_rate']:.1%} -> {after['success_rate']:.1%}  "
      f"(chance {task.chance:.0%})")
print(brain.cortex.weight_drift())

path = brain.save("checkpoints/cue.brain")
print(f"\nwrote {path} - the weights only, and it refuses to load onto "
      f"different wiring")
