"""
Reward, and what it does.

Two things happen when a brain is paid, and they are worth seeing separately:

  1. behaviour changes immediately - dopamine raises the approach gain and
     lowers exploration, before any synapse has moved
  2. every synapse that was recently active carries an eligibility trace, and
     the payment cashes those in

    python examples/02_reward_it.py
"""
import numpy as np

from cortex import Brain

brain = Brain.demo(n=2000, sim_steps=200, explore=2.0)

field = np.zeros((120, 160), dtype=np.float32)
field[50:70, 20:50] = 1.0

print("\n-- before any reward --")
brain.look(field)
print(f"  {brain.dopamine}")
print(f"  exploration sigma {brain.dopamine.explore_sigma(2.0):.2f}, "
      f"approach {brain.dopamine.approach_gain():.2f}")
print(f"  eligible synapses: {brain.plasticity.stats()['eligible_edges']:,}")

print("\n-- paid +1 --")
rpe = brain.reward(1.0)
print(f"  prediction error {rpe:+.3f}   {brain.dopamine}")
print(f"  exploration sigma {brain.dopamine.explore_sigma(2.0):.2f}, "
      f"approach {brain.dopamine.approach_gain():.2f}")
print(f"  synapses moved: {brain.cortex.weight_drift()['changed']:,}")

print("\n-- paid +1 twenty more times --")
for _ in range(20):
    brain.look(field)
    rpe = brain.reward(1.0)
print(f"  prediction error now {rpe:+.3f}")
print("  The same reward, over and over, stops teaching anything: a dopamine "
      "cell fires on surprise, not on reward. To keep learning, the task has "
      "to get harder or the payment has to change.")
