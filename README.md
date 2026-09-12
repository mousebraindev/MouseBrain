# mousecortex

A cubic millimetre of mouse visual cortex you can attach to anything, and train
with reward.

71,807 neurons. 2,338,483 signed connections. 7,296,357 synapses. Every one of
them measured from a real mouse by electron microscopy, and every cell's
preferred direction of motion measured from that same mouse while it watched
video.

```python
from cortex import Brain

brain = Brain.demo()          # runs now, no download, synthetic wiring
act = brain.look(field)       # any 2D array in, a decision out
brain.reward(+1.0)            # dopamine, and it learns from it
```

```bash
pip install -e .
python -m cortex.cli demo     # about 15 seconds
```

---

## What it is

A spiking network with a real connectome in it, wrapped so that four lines of
your code can use it. There is no framework here: a `Brain` takes a 2D array
and returns a decision, and a scalar reward is the only thing that goes back in.

**Four pieces, and you can use any of them alone.**

| | what it does | where |
|---|---|---|
| `Graph` | the wiring, measured or synthetic | `cortex/graph.py` |
| `Cortex` | leaky integrate-and-fire over that wiring | `cortex/core.py` |
| `Retina` + `Readout` | a 2D field in, a decision out | `cortex/field.py`, `cortex/readout.py` |
| `Dopamine` + `RewardModulatedSTDP` | reward changes behaviour, then synapses | `cortex/reward.py`, `cortex/plasticity.py` |

## It runs without the data

The measured connectome is 3 GB of public files. Nothing in this repo needs
them to start: `Graph.synthetic()` builds a network with the cube's statistics -
89% excitatory cells, log-normal synapse counts, local inhibition and longer
excitation, a retinotopic map with scatter, clustered direction preferences - so
code written against it runs unchanged against the real thing.

**The synthetic graph is generated, not measured.** It says so in
`graph.describe()`, in every checkpoint it produces, and here. No result from it
is evidence about a mouse. To get the real one, see [the data](#the-data).

## Use it on your own problem

A visual area accepts one thing: a field of numbers laid out in space. That is
less of a restriction than it sounds. A screenshot is a field. So is a game
board, an order book drawn as a heatmap, a spectrogram, an occupancy grid, or a
2D projection of an embedding. If you can draw it, this can look at it.

A `Task` is four methods:

```python
from cortex import Brain, Task, train

class MyTask(Task):
    def reset(self):            ...   # start an episode
    def observe(self):          ...   # -> a 2D numpy array
    def act(self, action):      ...   # do it, -> reward as a float
    def done(self):             ...   # -> bool

train(Brain.demo(), MyTask(), episodes=50)
```

The decision comes back as an `Action`: `dx, dy` (where to move), `commit`
(the discrete act - a click, a grasp), `choice` (one of k options), and
`confidence`. You choose which of them means anything for your problem.

Worked examples in `examples/`: a first look, what reward does, a full training
run, a task of your own, and a live web page.

## Reward, and what dopamine actually does here

Two separate things, and conflating them is how this gets oversold.

**It changes behaviour now.** Dopamine raises the approach gain and lowers
exploration noise within a single step, before any synapse has moved. A brain
that has just been paid stops casting about and commits. That part is immediate
and visible in `examples/02_reward_it.py`.

**It teaches, later.** No synapse knows whether the last second went well. Each
one keeps an eligibility trace - "I was recently part of something" - decaying
over about a second. Dopamine multiplies every trace at once, so the synapses
that were involved move and the rest do not:

```
dw/dt          = learning_rate * dopamine(t) * eligibility(t)
d(elig)/dt     = -elig / tau_e + STDP(pre, post)
```

That is the three-factor rule (Izhikevich 2007, *Solving the distal reward
problem*). Timing decides the sign of the trace; dopamine decides whether the
trace is worth anything.

**Dopamine fires on surprise, not on reward.** The burst is the prediction
error against a running baseline. Pay the same +1 often enough and the
expectation catches up and the signal goes quiet - which is what a real dopamine
cell does, and what `test_expected_reward_stops_teaching` asserts.

Four rules the learning keeps, each enforced by a test:

- **Dale's law.** An excitatory cell cannot learn its way into being
  inhibitory. Weights move in magnitude; the sign belongs to the cell.
- **A ceiling.** No synapse exceeds 3x the strength the microscope measured.
  Without it a long generous run turns the connectome into whatever the task
  wanted and nothing measured survives.
- **Synaptic scaling.** Each cell's total incoming weight is held where it was
  found, so learning is a competition between that cell's inputs rather than a
  volume knob on the whole network.
- **No dopamine, no change.** Activity on its own moves nothing.

## Training it yourself

```bash
python -m cortex.cli train --task cue --episodes 600 --save checkpoints/cue.brain
python -m cortex.cli eval  --task cue --load checkpoints/cue.brain
```

A checkpoint holds the weights and nothing else, and refuses to load onto a
different connectome - weights from one wiring mean nothing on another.

Evaluation freezes plasticity, so the number that comes out is what the brain
knows and not what it picked up while being measured.

### What training actually achieves, measured

Four seeds, the four-option cue task, 600 episodes, 100 evaluation episodes
before and after with plasticity frozen. Chance is 25%.

| seed | before | after |
|---|---|---|
| 0 | 29.0% | **47.0%** |
| 1 | 32.0% | **50.0%** |
| 2 | 3.0% | 8.0% |
| 3 | 16.0% | 21.0% |
| mean | 20.0% | 31.5% |

Every seed improved, by +11.5 points on average.

**The control.** The same protocol with plasticity frozen - evaluate, do
nothing, evaluate again - drifts by +1.0 points with a spread of 3.0. So the
two +18-point gains are five or six times the noise floor and are real; the two
+5-point gains are inside it and should be read as "did not get worse".

**Where it does not get to.** Two of the four seeds end above chance and two do
not. The motor pool is an arbitrary grouping of cells, so a brain can start with
a mapping that is actively wrong - seed 2 begins at 3%, which is not chance,
it is the decoder confidently picking the same wrong group - and 600 episodes is
not enough to dig out of that. The claim is "training reliably moves a brain up
from wherever it started", not "it solves the task".

**Two things had to be fixed before any of this worked**, and both were bugs
rather than tuning:

- The motor pool was being driven by injected spikes from outside the network,
  so its firing was the same whatever the network did and no synapse could earn
  credit for it. It is now held near threshold the way an awake cortical cell
  is, and what pushes it over is the input it receives.
- A global reward moved every eligible synapse the same way and the network
  simply got louder. Synaptic scaling turned that back into a competition
  between each cell's inputs.

**And one parameter mattered more than everything else.** Depression had to
stop cancelling potentiation: with A+/A- near 1 the eligibility a firing
population builds is close to net zero and reward has nothing to act on. Swept
on the isolated version of the task, the gain rises monotonically as
potentiation pulls ahead - ratio 0.95 gives +0.04, 3 gives +0.18, 10 gives
+0.22 - so the default is now 10, with synaptic scaling holding the runaway
that would otherwise cause. The learning rate is 1.5 mV per unit of
dopamine-times-eligibility, which is far above anything a real synapse does and
is set that way so a demo moves at all.

## The rails, when it touches the world

`ScreenTask` is the only part that acts outside the process, and it is optional.

- **No wallet, no keys, no stored session.** The browser gets a blank profile.
- **No keyboard.** It cannot type, so it cannot fill a field or answer a prompt.
- **An allowlist, not a keyword blocklist.** An empty allowlist is refused.
- **Every click checked before it lands.** Anything reading as submit, upload,
  sign-in, download, purchase or delete is vetoed.
- **An anti-stall budget.** If the cursor has not moved in N steps the view
  scrolls, so a dead end does not become a permanent home.

## Nothing personal ships

`tools/privacy_scan.py` runs over every file git would publish and fails on
absolute home paths, e-mail addresses, social handles, API keys, private keys,
JWTs, cloud credentials, wallet addresses, webhook URLs and private IPs. It runs
in the test suite and in CI, so it is not a promise anyone has to remember to
keep.

```bash
python tools/privacy_scan.py
```

`data/`, `build/`, `checkpoints/`, `.env`, `profile/` and `*.npz` are ignored:
nothing measured, trained or personal is committed.

## Four things the data did not want to do

**Inhibition was inverted.** The raw matrix has 1.67M inhibitory edges against
0.67M excitatory, in a volume that is 89% excitatory cells. That is not cortex,
that is reconstruction: interneuron axons are local and survive inside the cubic
millimetre, pyramidal axons leave it and get cut. Untouched, the network sits at
-74 mV, below its own resting potential. `balance.py` corrects it with one
scalar per cell toward the textbook 80/20 split.

**The decoder had an opinion before it saw anything.** 957 cells prefer 225-270
degrees and 361 prefer 135-180. Firing every cell at the same rate already
yields dx=-1.9 dy=+13.5, which would read as the cortex deciding to go
down-left. That offset is measured once and subtracted.

**Flat luminance drowned the signal.** Driving on absolute brightness left the
edge contribution at 0.7% of total drive - every cell inside a bright box got
the same input. Drive is now local contrast plus the image gradient projected
onto each cell's own preferred direction, half-wave rectified.

**The brain rebooted every frame.** With membrane state discarded between
control steps, direction autocorrelation at lag 1 was -0.076: independent draws.
Carrying state, and integrating over 80 ms instead of 20, brings it to +0.38.
A decoder cell fires 0.32 times in 20 ms and 79% of them are silent, so a vector
built from one short window is close to a coin flip.

## What is NOT real, stated plainly

- **The cursor is not goal-directed.** Untrained, direction autocorrelation past
  lag 1 sits at noise. It drifts along contrast; it does not go anywhere on
  purpose. Training moves it off that floor, and does not carry it to solving
  anything - see the numbers above.
- **It is cortex, so it has no motor output at all.** Nothing in this volume
  ever moved a mouse's paw. Every readout is a decode, not a command, and the
  motor pool is an arbitrary grouping of cells, like the placement of an
  electrode array.
- **The click is invented.** A fly clicks by stopping, on a stopping neuron.
  Cortex has no stopping neuron, so a commit here is the population having
  arrived: firing hard, pulled nowhere. That is a modelling choice made by a
  person.
- **Several constants are chosen, not measured.** 0.40 mV per synapse (the fly
  has a measured 0.275; cortex has no agreed figure), the 4:1 E/I target, the
  learning rate - which is far above anything a real synapse does, and set that
  way so a demo shows movement at all. Every one of them is labelled MEASURED or
  CHOSEN in `cortex/params.py`, with the reasoning next to it.
- **The digital twin is a model.** Receptive fields and direction tuning come
  from a network trained on real recordings. The wiring is measured; the tuning
  is predicted from measurements.
- **The learning rule is not the mouse's.** Reward-modulated STDP is a model of
  how dopamine gates plasticity, not something this connectome tells you.

## The data

Public, no account and no key.

```
storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1412/
  connections_with_nuclei.csv.gz                3.0 GB
  aibs_metamodel_celltypes_v661_merged.csv.gz   3.5 MB
  functional_properties_v3_bcm_merged.csv.gz    746 KB
  coregistration_manual_v4_merged.csv.gz        942 KB
bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/
  digital_twin_properties/v2/readout/readout_locations.npy
  digital_twin_properties/v2/readout/units.csv
  digital_twin_properties/v2/ori_dir_tuning/units.csv
  digital_twin_properties/v2/anatomy/units.csv
  digital_twin_properties/v2/performance/units.csv
```

Put them under `data/` (the exact layout is printed by `cortex build`), then:

```bash
pip install -e ".[data]"
python -m cortex.build          # -> build/graph.npz
python -m cortex.cli demo --measured
```

71,807 neurons and 2,338,483 signed edges come out. If your numbers differ, one
of us has a bug.

Measured from the volume, not assumed: it spans 1380.7 x 796.2 x 502.1 µm, and
median soma depth runs L2/3 182.8, L4 320.2, L5 433.1, L6 632.4 µm.

## Install

```bash
pip install -e .                  # numpy and scipy, nothing else
pip install -e ".[data]"          # + pandas, to rebuild the connectome
pip install -e ".[screen]"        # + pillow and playwright, for a live page
pip install -e ".[dev]" && pytest -q
```

## Credits

Connectome and functional data from the MICrONS Consortium, released for open
use. Simulation approach after Shiu et al. 2024. Digital twin properties from
Wang et al. 2025 and Ding, Fahey, and Papadopoulos et al. 2025. The learning
rule follows Izhikevich 2007; synaptic scaling follows Turrigiano. Not
affiliated with any of them.

MIT licensed.
