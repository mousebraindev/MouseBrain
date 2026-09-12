"""
Every constant in one file, each one labelled measured or chosen.

The split matters more than the values. A number marked MEASURED comes out of
the data and moving it is a bug. A number marked CHOSEN is a modelling decision
made by a person; moving it is tuning, and the reason it sits where it sits is
written next to it.
"""
from dataclasses import dataclass


# ---------------------------------------------------------------- membrane ---
@dataclass
class Passive:
    """Leaky integrate-and-fire, cortical pyramidal values from the textbook."""
    v_rest: float = -70.0     # mV   MEASURED (standard cortical resting potential)
    v_thresh: float = -50.0   # mV   MEASURED
    v_reset: float = -65.0    # mV   MEASURED
    tau_m: float = 20.0       # ms   MEASURED
    refractory: float = 2.0   # ms   MEASURED
    dt: float = 0.2           # ms   CHOSEN: 100x smaller than tau_m, cheap enough


# ------------------------------------------------------------------ wiring ---
# CHOSEN. The fly connectome has a fitted 0.275 mV per synapse (Shiu et al.
# 2024). Cortex has no agreed figure; unitary EPSPs in mouse V1 run a few tenths
# of a mV. This is the single largest free knob in the whole model.
MV_PER_SYNAPSE = 0.40

# CHOSEN. Pairs joined by a single synapse are dominated by detection noise.
MIN_SYNAPSES = 2

# CHOSEN. The textbook cortical split: 80% of synaptic weight excitatory.
# Needed because the volume is a cube: interneuron axons stay inside it and
# survive, pyramidal axons leave it and get cut, so the raw matrix has 1.67M
# inhibitory edges against 0.67M excitatory in a volume that is 89% excitatory
# cells. Untouched, the network sits below its own resting potential.
EI_TARGET = 4.0


# --------------------------------------------------------------- perception ---
# CHOSEN drive scaling. A V1 cell answers to contrast, not to the absolute light
# level, so the power goes to edges and flat luminance is kept only as a weak
# "there is something here" signal.
SPONT_HZ = 2.5        # CHOSEN: V1 is never silent; without it a blank patch
                      # means zero drive, a still cursor, an unchanging image,
                      # and the run is over.
LUMINANCE_HZ = 40.0   # CHOSEN
CONTRAST_HZ = 180.0   # CHOSEN
EDGE_HZ = 140.0       # CHOSEN
GRAD_PX = 6           # CHOSEN: gradient window, in field pixels


# ------------------------------------------------------------------ readout ---
CC_MIN = 0.3          # CHOSEN: keep twin units that predict the real cell at
                      # r >= 0.3, the threshold the dataset's own papers use
GDSI_MIN = 0.1        # CHOSEN: below this a cell has no usable direction
QUIET_HZ = 0.4        # CHOSEN: under this the population is silent and a silent
                      # population has to mean a still output, not a confident
                      # push in the direction of the sampling bias
SMOOTH = 0.25         # CHOSEN: first-order filter on the output, because a
                      # motor system integrates too - muscle does not follow
                      # spikes one for one
APPROACH = 0.62       # CHOSEN: fraction of the remaining gap covered per step
SIM_STEPS = 400       # CHOSEN: 400 * 0.2 ms = 80 ms of brain time per decision.
                      # At 20 ms a decoder cell fires 0.32 times and 79% are
                      # silent, so the vector is coin flips: measured
                      # repeatability r=1.53 at 20 ms, 2.70 at 160 ms.


# -------------------------------------------------------------- dopamine -----
# Reward-modulated STDP, the three-factor rule (Izhikevich 2007, "Solving the
# distal reward problem"). All CHOSEN: this is the learning machinery bolted on
# to the measured wiring, not something the connectome tells you.
TAU_PRE = 20.0        # ms, presynaptic spike trace
TAU_POST = 20.0       # ms, postsynaptic spike trace
TAU_ELIGIBILITY = 1000.0   # ms. The gap a reward has to reach back across.
TAU_DOPAMINE = 200.0       # ms, phasic burst decay
A_PLUS = 0.020        # pre-before-post, potentiation
A_MINUS = 0.002       # post-before-pre, depression. Ten times smaller than
                      # A_PLUS, which is the opposite of the usual choice and
                      # was arrived at by measurement, not taste. With the two
                      # near-balanced, the eligibility a firing population
                      # builds is close to net zero and reward has almost
                      # nothing to act on: swept over the cue task, the gain
                      # rises monotonically as potentiation pulls ahead
                      # (A+/A- of 0.95 -> +0.04, 3 -> +0.18, 10 -> +0.22).
                      # What normally makes a rule this LTP-heavy blow up is
                      # runaway potentiation, and synaptic scaling below is
                      # what holds it: totals are fixed, so the only thing
                      # potentiation can do is redistribute.
LEARNING_RATE = 1.5   # mV per unit of (dopamine * eligibility). Far above
                      # anything a real synapse does, and chosen that way: at a
                      # biological rate the demos would need tens of thousands
                      # of episodes to move at all. Measured on the four-option
                      # cue task over 600 episodes, 0.3 gives +0.06 and 1.5
                      # gives +0.22 on the isolated version of the task.
W_MAX_FACTOR = 3.0    # a synapse may not grow past 3x its measured strength
DA_TONIC = 0.0        # baseline dopamine with no reward prediction error
DA_MAX = 4.0          # saturation, so one huge reward cannot rewrite the brain
RPE_TAU = 40.0        # steps. Running baseline of expected reward: the brain
                      # learns from surprise, not from reward. A reward you
                      # always get stops teaching you anything.
