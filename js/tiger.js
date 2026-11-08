/* ============================================================================
 * The classic "Tiger" POMDP (Kaelbling, Littman & Cassandra, 1998)
 * and an exact finite-horizon-style infinite-horizon solver on a belief grid,
 * used as ground truth to compare POMCP against.
 *
 *   Two doors.  A tiger is behind one of them (uniform prior).
 *   listen      : reward -1, hear the tiger's side correctly with prob. p
 *   open-left   : +10 if the tiger is on the right, -100 if on the left
 *   open-right  : +10 if the tiger is on the left,  -100 if on the right
 *
 * Opening a door ends the episode.  Optimal behaviour is to listen until the
 * belief is sharp enough, then open the opposite door.
 * ==========================================================================*/
(function (global) {
  'use strict';

  class TigerEnv {
    constructor(opts = {}) {
      this.noise = opts.noise ?? 0.85;       // P(correct | listen)
      this.gamma = opts.gamma ?? 0.95;
      this.rCorrect = opts.rCorrect ?? 10;
      this.rWrong = opts.rWrong ?? -100;
      this.listenCost = opts.listenCost ?? -1;
      this.actions = [
        { id: 'listen', label: 'Listen', short: 'listen' },
        { id: 'open-left', label: 'Open left door', short: 'open L' },
        { id: 'open-right', label: 'Open right door', short: 'open R' },
      ];
    }

    sampleInitialState(rng) {
      return { tiger: rng.next() < 0.5 ? 'L' : 'R', done: false };
    }

    transition(state, action, rng) {
      if (state.done) return { state, reward: 0, done: true };
      if (action === 'listen') {
        return {
          state: { tiger: state.tiger, done: false },
          reward: this.listenCost,
          done: false,
        };
      }
      const openedLeft = action === 'open-left';
      const correct = (openedLeft && state.tiger === 'R') ||
                      (!openedLeft && state.tiger === 'L');
      return {
        state: { tiger: state.tiger, done: true },
        reward: correct ? this.rCorrect : this.rWrong,
        done: true,
      };
    }

    sampleObservation(state, action, rng) {
      if (action !== 'listen' || state.done) return 'none';
      const correct = rng.next() < this.noise;
      const heardLeft = state.tiger === 'L' ? correct : !correct;
      return heardLeft ? 'L' : 'R';
    }

    observationProbability(state, action, observation) {
      if (action !== 'listen' || state.done) return observation === 'none' ? 1 : 0;
      const heardLeft = observation === 'L';
      const correct = (heardLeft && state.tiger === 'L') ||
                      (!heardLeft && state.tiger === 'R');
      return correct ? this.noise : 1 - this.noise;
    }

    step(state, action, rng) {
      const tr = this.transition(state, action, rng);
      const observation = this.sampleObservation(tr.state, action, rng);
      return { state: tr.state, observation, reward: tr.reward, done: tr.done };
    }

    isTerminal(state) {
      return !!(state && state.done);
    }

    /** Exact Bayesian belief update for the Tiger problem. */
    bayesUpdate(b, action, observation) {
      if (action !== 'listen') return b;
      const p = this.noise;
      const likeLeft = observation === 'L' ? p : 1 - p;
      const likeRight = observation === 'L' ? 1 - p : p;
      const num = b * likeLeft;
      const den = num + (1 - b) * likeRight;
      return den > 1e-15 ? num / den : b;
    }
  }

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  /**
   * Solve the Tiger POMDP exactly on a uniform discretisation of the belief
   * simplex.  The Bellman backup is a contraction, so value iteration on the
   * 1-D belief grid converges quickly (linear interpolation between points).
   */
  function solveExact(env, opts = {}) {
    const points = opts.points ?? 501;
    const maxIter = opts.maxIter ?? 2000;
    const tol = opts.tol ?? 1e-11;
    const gamma = env.gamma;
    const p = env.noise;
    const n = points;
    const last = n - 1;

    const update = (b, obs) => {
      const likeLeft = obs === 'L' ? p : 1 - p;
      const likeRight = obs === 'L' ? 1 - p : p;
      const num = b * likeLeft;
      const den = num + (1 - b) * likeRight;
      return den > 1e-15 ? num / den : b;
    };

    const interp = (arr, b) => {
      const x = clamp(b, 0, 1) * last;
      const i = Math.floor(x);
      if (i >= last) return arr[last];
      const f = x - i;
      return arr[i] * (1 - f) + arr[i + 1] * f;
    };

    let V = new Float64Array(n);
    let iterations = 0;

    for (let iter = 0; iter < maxIter; iter++) {
      const next = new Float64Array(n);
      let diff = 0;
      for (let i = 0; i < n; i++) {
        const b = i / last;
        const openLeft = env.rWrong * b + env.rCorrect * (1 - b);
        const openRight = env.rCorrect * b + env.rWrong * (1 - b);
        const pLeft = p * b + (1 - p) * (1 - b);
        const bL = update(b, 'L');
        const bR = update(b, 'R');
        const listen = env.listenCost +
          gamma * (pLeft * interp(V, bL) + (1 - pLeft) * interp(V, bR));
        const v = Math.max(openLeft, openRight, listen);
        next[i] = v;
        const d = Math.abs(v - V[i]);
        if (d > diff) diff = d;
      }
      V = next;
      iterations = iter + 1;
      if (diff < tol) break;
    }

    const valuesAt = (b) => {
      b = clamp(b, 0, 1);
      const openLeft = env.rWrong * b + env.rCorrect * (1 - b);
      const openRight = env.rCorrect * b + env.rWrong * (1 - b);
      const pLeft = p * b + (1 - p) * (1 - b);
      const listen = env.listenCost +
        gamma * (pLeft * interp(V, update(b, 'L')) +
                 (1 - pLeft) * interp(V, update(b, 'R')));
      let best = 'listen';
      let bestV = listen;
      if (openLeft > bestV) { best = 'open-left'; bestV = openLeft; }
      if (openRight > bestV) { best = 'open-right'; bestV = openRight; }
      return { openLeft, openRight, listen, best, bestV, value: interp(V, b) };
    };

    // Where does it become optimal to open?  By symmetry, opening the left
    // door is optimal for b < 1 - threshold.
    let threshold = 1;
    const scan = 2001;
    for (let i = 0; i <= scan; i++) {
      const b = 0.5 + (0.5 * i) / scan;
      if (valuesAt(b).best === 'open-right') {
        threshold = b;
        break;
      }
    }

    return {
      V,
      points,
      iterations,
      interp,
      valuesAt,
      threshold,
      valueAt(b) {
        return interp(V, b);
      },
      /** sample the value curve for plotting */
      curve(samples = 201) {
        const out = new Array(samples);
        for (let i = 0; i < samples; i++) {
          const b = i / (samples - 1);
          out[i] = { b, v: interp(V, b) };
        }
        return out;
      },
    };
  }

  const api = { TigerEnv, solveExact };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.Tiger = api;
})(typeof window !== 'undefined' ? window : globalThis);
