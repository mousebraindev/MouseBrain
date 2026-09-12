"""
The smallest thing that works: show a brain a picture, see what it decides.

    python examples/01_hello_brain.py

No download, no browser, no GPU. Synthetic wiring, so this is a demonstration
of the machinery and not evidence about a mouse.
"""
import numpy as np

from cortex import Brain

brain = Brain.demo(n=2000, sim_steps=200)

# any 2D array will do: a screenshot, a board, a heatmap, a spectrogram
field = np.zeros((120, 160), dtype=np.float32)
field[30:50, 110:140] = 1.0        # something bright, up and to the right

print(f"\none decision takes {brain.window_ms:.0f} ms of brain time\n")
for i in range(6):
    act = brain.look(field)
    print(f"  {i}  {act}   {act.detail['spikes_per_sec']:8.0f} spikes/s")

print("\nNothing has been rewarded yet, so nothing has been learned. The vector "
      "is a decode of where the population is looking, not a plan.")
