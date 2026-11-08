/* Node test suite for the browser-side POMCP core.
 *
 *     node --test test/
 */
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { RNG, Planner, ParticleBelief } = require('../js/pomcp-core');
const { TigerEnv, solveExact } = require('../js/tiger');

function certainBelief(env, rng, count, tiger) {
  const belief = new ParticleBelief(env, count, rng);
  belief.particles = new Array(count).fill(null).map(() => ({ tiger, done: false }));
  return belief;
}

test('Tiger rewards and terminal states', () => {
  const env = new TigerEnv();
  const rng = new RNG(1);
  let tr = env.transition({ tiger: 'L', done: false }, 'listen', rng);
  assert.equal(tr.reward, -1);
  assert.equal(tr.done, false);

  tr = env.transition({ tiger: 'L', done: false }, 'open-left', rng);
  assert.equal(tr.reward, -100);
  assert.equal(tr.done, true);

  tr = env.transition({ tiger: 'L', done: false }, 'open-right', rng);
  assert.equal(tr.reward, 10);
  assert.equal(tr.done, true);
});

test('Bayesian belief update is symmetric', () => {
  const env = new TigerEnv({ noise: 0.9 });
  assert.ok(Math.abs(env.bayesUpdate(0.5, 'listen', 'L') - 0.9) < 1e-12);
  assert.ok(Math.abs(env.bayesUpdate(0.5, 'listen', 'R') - 0.1) < 1e-12);
  assert.ok(Math.abs(env.bayesUpdate(0.9, 'listen', 'L') - 0.987804878) < 1e-8);
});

test('particle filter follows the exact posterior', () => {
  const env = new TigerEnv({ noise: 0.9 });
  const rng = new RNG(5);
  const belief = new ParticleBelief(env, 20000, rng);
  // Deliberately ordered particles: a good resampler must not be fooled.
  belief.particles = new Array(20000).fill(null).map((_, i) => ({
    tiger: i % 2 ? 'L' : 'R',
    done: false,
  }));
  belief.update('listen', 'L');
  assert.ok(Math.abs(belief.fractionTigerLeft() - 0.9) < 0.015);
});

test('exact solver matches the known default benchmark', () => {
  const env = new TigerEnv({ noise: 0.9, gamma: 0.95 });
  const exact = solveExact(env, { points: 501 });
  assert.ok(Math.abs(exact.threshold - 0.9805) < 0.005);
  assert.ok(Math.abs(exact.valueAt(0.5) - 5.322) < 0.05);
  assert.equal(exact.valuesAt(0.5).best, 'listen');
  assert.equal(exact.valuesAt(0.01).best, 'open-left');
  assert.equal(exact.valuesAt(0.99).best, 'open-right');
});

test('POMCP listens when the belief is uncertain', () => {
  const env = new TigerEnv({ noise: 0.9 });
  const rng = new RNG(11);
  const planner = new Planner(env, {
    simulations: 5000,
    exploration: 100,
    gamma: 0.95,
    rng,
  });
  const belief = new ParticleBelief(env, 1200, rng);
  const plan = planner.plan(belief);
  assert.equal(plan.action, 'listen');
});

test('POMCP opens the safe door when the state is certain', () => {
  const env = new TigerEnv({ noise: 0.9 });
  const rng = new RNG(12);
  const planner = new Planner(env, {
    simulations: 5000,
    exploration: 100,
    gamma: 0.95,
    rng,
  });

  let belief = certainBelief(env, rng, 1000, 'L');
  planner.reset();
  assert.equal(planner.plan(belief).action, 'open-right');

  belief = certainBelief(env, rng, 1000, 'R');
  planner.reset();
  assert.equal(planner.plan(belief).action, 'open-left');
});

test('tree reuse advances along the received history', () => {
  const env = new TigerEnv({ noise: 0.9 });
  const rng = new RNG(13);
  const planner = new Planner(env, {
    simulations: 3000,
    exploration: 100,
    gamma: 0.95,
    rng,
  });
  const belief = new ParticleBelief(env, 1000, rng);
  planner.plan(belief);
  const before = planner.root;
  planner.advance('listen', 'L');
  assert.notEqual(planner.root, before);
  assert.ok(planner.root.N > 0);
});
