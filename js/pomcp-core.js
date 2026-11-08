/* ============================================================================
 * POMCP — Partially Observable Monte-Carlo Planning
 * A compact, dependency-free implementation of Silver & Veness (NeurIPS 2010).
 *
 * The planner is completely generic: an environment only needs
 *
 *   actions                        -> [{ id, label, short? }, ...]
 *   sampleInitialState(rng)        -> s
 *   transition(s, a, rng)          -> { state, reward, done }
 *   sampleObservation(s, a, rng)   -> o            (s is the *next* state)
 *   observationProbability(s,a,o)  -> P(o | s, a)  (s is the *next* state)
 *   isTerminal(s)                  -> bool
 *
 * i.e. a generative model plus an observation likelihood.  The likelihood is
 * only used by the particle filter, never by the tree search itself.
 * ==========================================================================*/
(function (global) {
  'use strict';

  /* ------------------------------- RNG ---------------------------------- */
  class RNG {
    constructor(seed = 0x9e3779b9) {
      this.s = seed >>> 0;
    }
    next() {
      // mulberry32
      this.s = (this.s + 0x6d2b79f5) >>> 0;
      let t = this.s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    int(n) {
      return Math.floor(this.next() * n);
    }
    pick(arr) {
      return arr[this.int(arr.length)];
    }
    bernoulli(p) {
      return this.next() < p;
    }
  }

  /* ------------------------------ Tree node ------------------------------ */
  class Node {
    constructor() {
      this.visited = false;
      this.N = 0;                 // visits to this history
      this.Na = new Map();        // action -> visit count
      this.Qa = new Map();        // action -> mean return
      this.children = new Map();  // "action|observation" -> Node
    }
  }

  /* --------------------------- Particle belief -------------------------- */
  /**
   * Sequential importance resampling filter.
   * Given the previous particle set (a sample of b), an action and the
   * observation that was actually received, produce a sample of b'.
   *
   *   b'(s') ∝ O(o | s', a) · Σ_s T(s' | s, a) b(s)
   *
   * Particles are propagated through the transition model first, then
   * reweighted by the observation likelihood and resampled.
   */
  class ParticleBelief {
    constructor(env, count, rng) {
      this.env = env;
      this.rng = rng;
      this.particles = new Array(count);
      for (let i = 0; i < count; i++) {
        this.particles[i] = env.sampleInitialState(rng);
      }
    }

    get size() {
      return this.particles.length;
    }

    sample(rng = this.rng) {
      return this.particles[rng.int(this.particles.length)];
    }

    /** Fraction of particles in the "left" state (demo-specific helper). */
    fractionTigerLeft() {
      let n = 0;
      for (const p of this.particles) if (p.tiger === 'L') n++;
      return n / this.particles.length;
    }

    update(action, observation) {
      const env = this.env;
      const rng = this.rng;
      const n = this.particles.length;
      const states = new Array(n);
      const weights = new Float64Array(n);
      let maxWeight = 0;

      for (let i = 0; i < n; i++) {
        const tr = env.transition(this.particles[i], action, rng);
        const s2 = tr.state;
        const w = env.observationProbability(s2, action, observation);
        states[i] = s2;
        weights[i] = w > 0 ? w : 0;
        if (weights[i] > maxWeight) maxWeight = weights[i];
      }

      // If every particle considers the observation impossible, fall back to
      // an unweighted propagation (particle impoverishment).
      const next = new Array(n);
      if (maxWeight <= 0) {
        for (let i = 0; i < n; i++) next[i] = states[i];
      } else {
        // Systematic resampling — lower variance than multinomial sampling.
        // Shuffle first: if particles are ordered by state, a fixed
        // systematic-sampling offset can persistently skip a minority group.
        for (let i = n - 1; i > 0; i--) {
          const j = rng.int(i + 1);
          const ts = states[i];
          states[i] = states[j];
          states[j] = ts;
          const tw = weights[i];
          weights[i] = weights[j];
          weights[j] = tw;
        }

        let total = 0;
        for (let i = 0; i < n; i++) {
          weights[i] /= maxWeight;
          total += weights[i];
        }
        const step = total / n;
        let u = rng.next() * step;
        let acc = 0;
        let j = 0;
        for (let i = 0; i < n; i++) {
          while (j < n - 1 && u > acc + weights[j]) {
            acc += weights[j];
            j++;
          }
          next[i] = states[j];
          u += step;
        }
      }
      this.particles = next;
      return this;
    }
  }

  /* ------------------------------- Planner ------------------------------ */
  class Planner {
    /**
     * @param {object} env     generative POMDP model (see header)
     * @param {object} opts    { simulations, exploration, gamma, maxDepth, rng }
     */
    constructor(env, opts = {}) {
      this.env = env;
      this.simulations = opts.simulations ?? 2000;
      this.exploration = opts.exploration ?? 25;
      this.gamma = opts.gamma ?? 0.95;
      this.maxDepth = opts.maxDepth ?? 24;
      this.initVisits = opts.initVisits ?? 1;
      this.rng = opts.rng ?? new RNG(12345);
      this.root = new Node();
      this.nodeCount = 0;
      this.totalSimulations = 0;
    }

    reset() {
      this.root = new Node();
      this.nodeCount = 0;
      this.totalSimulations = 0;
    }

    /** Reuse the subtree that matches the action/observation just received. */
    advance(action, observation) {
      const child = this.root.children.get(action + '|' + observation);
      this.root = child || new Node();
      this.nodeCount = child ? this.nodeCount : 0;
    }

    /** UCT action selection at a node that has been visited at least once. */
    _select(node) {
      const actions = this.env.actions;
      let best = actions[0].id;
      let bestScore = -Infinity;
      const logN = Math.log(Math.max(1, node.N));

      // Force a minimum number of visits per action.  This makes the planner
      // robust to a single unlucky rollout (important when rewards have a wide
      // range, e.g. -100/ +10).
      if (this.initVisits > 1) {
        let least = Infinity;
        let candidate = null;
        for (const a of actions) {
          const na = node.Na.get(a.id) || 0;
          if (na < this.initVisits && na < least) {
            least = na;
            candidate = a.id;
          }
        }
        if (candidate !== null) return candidate;
      }

      for (let i = 0; i < actions.length; i++) {
        const id = actions[i].id;
        const na = node.Na.get(id) || 0;
        if (na === 0) return id; // every action is tried at least once
        const exploit = node.Qa.get(id);
        const explore = this.exploration * Math.sqrt(logN / na);
        const score = exploit + explore;
        if (score > bestScore) {
          bestScore = score;
          best = id;
        }
      }
      return best;
    }

    /**
     * One Monte-Carlo simulation.
     * Returns the discounted return from `state` under the current tree policy.
     */
    _simulate(state, node, depth) {
      if (depth >= this.maxDepth || this.env.isTerminal(state)) return 0;

      if (!node.visited) {
        node.visited = true;
        this.nodeCount++;
        return this._rollout(state, depth);
      }

      const action = this._select(node);
      const tr = this.env.transition(state, action, this.rng);
      const observation = this.env.sampleObservation(tr.state, action, this.rng);
      let ret;

      if (tr.done) {
        ret = tr.reward;
      } else {
        const key = action + '|' + observation;
        let child = node.children.get(key);
        if (!child) {
          child = new Node();
          node.children.set(key, child);
        }
        ret = tr.reward + this.gamma * this._simulate(tr.state, child, depth + 1);
      }

      node.N++;
      const na = (node.Na.get(action) || 0) + 1;
      node.Na.set(action, na);
      const oldQ = node.Qa.get(action) || 0;
      node.Qa.set(action, oldQ + (ret - oldQ) / na);
      return ret;
    }

    /** Random rollout used to initialise a newly-created node. */
    _rollout(state, depth) {
      this.env.onRolloutStart?.();
      let total = 0;
      let discount = 1;
      const actions = this.env.actions;

      for (let d = depth; d < this.maxDepth; d++) {
        if (this.env.isTerminal(state)) break;
        const action = actions[this.rng.int(actions.length)].id;
        const tr = this.env.transition(state, action, this.rng);
        total += discount * tr.reward;
        discount *= this.gamma;
        state = tr.state;
        if (tr.done) break;
      }
      return total;
    }

    /**
     * Build a search tree from the current belief and return the recommended
     * action together with root statistics for visualisation.
     *
     * A fresh state is sampled from the belief for every simulation: this is
     * what turns the tree value into a belief-space value rather than the
     * value of one particular hidden state.
     */
    plan(belief, simulations = this.simulations) {
      const started = Date.now();
      const accepted = [];

      for (let i = 0; i < simulations; i++) {
        const state = belief.sample(this.rng);
        // Root particles can occasionally be terminal if the caller reuses a
        // belief; skip those rather than wasting a simulation.
        if (this.env.isTerminal(state)) continue;
        this._simulate(state, this.root, 0);
        accepted.push(state);
      }
      this.totalSimulations += simulations;

      const rows = this.env.actions.map((a) => ({
        id: a.id,
        short: a.short || a.label || a.id,
        label: a.label || a.id,
        N: this.root.Na.get(a.id) || 0,
        Q: this.root.Qa.get(a.id) ?? null,
        visited: this.root.Na.has(a.id),
      }));

      // Greedy action w.r.t. the estimated Q-values (ties -> more visits).
      let best = null;
      for (const row of rows) {
        if (!row.visited) continue;
        if (
          best === null ||
          row.Q > best.Q + 1e-9 ||
          (Math.abs(row.Q - best.Q) <= 1e-9 && row.N > best.N)
        ) {
          best = row;
        }
      }
      if (!best) best = rows[0];

      return {
        action: best.id,
        rows,
        nodes: this._countNodes(this.root),
        depth: this._treeDepth(this.root, 0),
        simulations,
        ms: Date.now() - started,
        rootN: this.root.N,
        root: this.root,
        beliefSamples: accepted.length,
      };
    }

    _countNodes(node) {
      let n = 1;
      for (const child of node.children.values()) n += this._countNodes(child);
      return n;
    }

    _treeDepth(node, depth) {
      let best = depth;
      for (const child of node.children.values()) {
        const d = this._treeDepth(child, depth + 1);
        if (d > best) best = d;
      }
      return best;
    }
  }

  const api = { RNG, Node, ParticleBelief, Planner };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.POMCP = api;
})(typeof window !== 'undefined' ? window : globalThis);
