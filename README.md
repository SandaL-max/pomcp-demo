# POMCP — online planning under partial observability

An interactive demo of **Partially Observable Monte-Carlo Planning** (POMCP)
on the classic **Tiger** POMDP, plus a small, dependency-free Python reference
implementation of the same algorithm.

POMCP searches over *histories* rather than states. A particle filter represents
what the agent believes, a generative model supplies sampled futures, and UCT
grows a tree over action–observation histories. The result is an anytime planner
for large POMDPs where exact belief-state planning is intractable.

---

## Quick start

### Browser demo (no build step)

Open `index.html` directly, or serve the folder:

```bash
cd pomcp-demo
python3 -m http.server 8000
# visit http://localhost:8000
```

The page is fully self-contained: plain HTML/CSS/JavaScript, no packages, no
network requests. Classic `<script>` tags are used so it also works from
`file://`.

### Python reference

```bash
cd pomcp-demo/python

python3 pomcp.py                          # 20 episodes, 5,000 sims/step
python3 pomcp.py --episodes 5 --trace     # print every decision and Q-value
python3 pomcp.py --sweep                  # compare simulation budgets
python3 pomcp.py --noise 0.65             # make listening nearly useless

python3 -m unittest discover -s python    # from the project root
```

Requires Python 3.9+ and only the standard library.

### Tests

```bash
python3 -m unittest discover -s python   # 14 Python tests
node --test test/pomcp.test.js            # 7 JavaScript tests (Node 18+)
```

Both suites check the belief update, particle filter, exact solver, and planner
behaviour, and they run in well under a second.

---

## What the demo shows

| Panel | What it contains |
| --- | --- |
| **Tiger problem** | Two hidden doors. Listening costs 1 and is right 90% of the time; opening ends the episode with +10 (safe) or −100 (tiger). |
| **Belief state** | The exact Bayesian posterior and the particle-filter estimate, with the exact “open” threshold marked on the bar. The dots are a sample of the particle cloud. |
| **Inside the search** | Root action values `Q(h,a)` and visit counts from POMCP, plus a depth-2 view of the history tree. Observation children show the conditional plan after hearing left/right. |
| **Exact belief-state planner** | Ground truth from value iteration on a 501-point belief grid: the optimal action and one-step values at the current belief, plus the value curve `V*(b)`. |
| **Run statistics** | Episode returns, safe-open rate, listens per episode, POMCP/exact action agreement, total simulations, and a return chart with the exact policy’s simulated average. |
| **Settings** | Simulations per step, UCT exploration constant, listen accuracy, and discount. Noise/discount changes restart the experiment. |

### Things worth trying

- Drop **simulations** to 500: quick decisions start disagreeing with the exact
  planner, and POMCP sometimes opens too early. That is the approximation error
  made visible.
- Raise **listen accuracy** to 95%: the belief sharpens faster and the planner
  opens after fewer listens.
- Lower **listen accuracy** to 65%: the optimal threshold moves, and the value
  of information drops.
- Toggle **Auto play**, or use `space`, `S`, and `N` for play/pause, step, and a
  new episode.

The expensive part is the online search: each step throws the previous plan
away in favour of a new one (keeping the matching subtree), which is exactly the
*online* part of online planning.

---

## Algorithm notes

### Belief update

Particles are propagated through the transition model, reweighted by the
observation likelihood, and resampled:

```
b'(s') ∝ O(o | s', a) Σ_s T(s' | s, a) b(s)
```

The implementation uses **systematic resampling**, with a shuffle before
sampling so a fixed particle ordering cannot persistently skip a minority group.

### Tree search

Each node is a history `h`. UCT selects actions:

```
a* = argmax_a [ Q(h,a) + c · sqrt( log N(h) / N(h,a) ) ]
```

and newly-created nodes are evaluated with a random rollout. Every simulation
samples a fresh start state from the particle belief, so the root values average
over the current uncertainty:

```
for each simulation:
    s ~ B(h)
    Simulate(s, h, 0)
return argmax_a Q(h,a)
```

After executing an action and receiving an observation, the planner follows the
corresponding child (tree reuse) and keeps planning from there.

### Exact benchmark

The Tiger POMDP is small enough to solve on a belief grid. The demo performs
value iteration over `b ∈ [0,1]` with linear interpolation between 501 points:

```
V(b) = max( OpenLeft(b), OpenRight(b),
            -1 + γ · E_o[ V( update(b,o) ) ] )
```

The resulting threshold, values, and curve are used as ground truth for
comparison. For the default parameters (listen accuracy 0.90, discount 0.95):

- switch threshold: `P(tiger left) = 0.9805`
- `V*(b = 0.5) = 5.322`

### Why POMCP sometimes gets it wrong

POMCP is an anytime Monte-Carlo method. With finite simulations, the value of
information is underestimated, so the planner can commit too early. The demo
keeps this visible instead of hiding it: the comparison badge, the exact
threshold, and the “opened at belief …” outcome all point at the gap.

A 300-episode JavaScript benchmark at the default settings gave:

| simulations | mean return | safe opens | exact-action match |
| ---: | ---: | ---: | ---: |
| 500 | −1.76 | 90.3% | 57.6% |
| 5,000 | +7.10 | 99.7% | 100.0% |

(These are empirical numbers; individual runs vary.)

---

## Files

```
pomcp-demo/
├── index.html              interactive demo
├── css/style.css           layout and styling
├── js/pomcp-core.js        generic POMCP planner + particle filter
├── js/tiger.js             Tiger model + exact belief-grid solver
├── js/app.js               UI controller and visualisations
├── python/pomcp.py         reference implementation + CLI
├── python/test_pomcp.py    Python unit tests
├── test/pomcp.test.js      JavaScript unit tests
└── README.md
```

The JavaScript planner is environment-agnostic. An environment supplies:

```js
actions
sampleInitialState(rng)
transition(state, action, rng)          // -> { state, reward, done }
sampleObservation(state, action, rng)   // state is the *next* state
observationProbability(state, action, o)
isTerminal(state)
```

The observation likelihood is used only by the particle filter, not by the tree
search, matching the generative-model interface of the original algorithm.

---

## References

- Silver, D. & Veness, J. (2010).
  [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps).
  *NeurIPS*.
- Kaelbling, L. P., Littman, M. L. & Cassandra, A. R. (1998).
  [Planning and acting in partially observable stochastic domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf).
  *Artificial Intelligence*, 101, 99–134.
- Sunberg, Z. & Kochenderfer, M. J. (2018).
  [Online algorithms for POMDPs with continuous state, action, and observation spaces](https://arxiv.org/abs/1709.06196).

This is a teaching demo, not a benchmark implementation. The Tiger problem is
made episodic (opening ends the episode), which keeps the interaction readable
while preserving the information-versus-commitment trade-off.
