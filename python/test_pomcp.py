"""Tests for the Python POMCP reference implementation.

Run from the project root:

    python3 -m unittest discover -s python -v
"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from pomcp import (  # noqa: E402
    LEFT,
    LISTEN,
    OPEN_LEFT,
    OPEN_RIGHT,
    RIGHT,
    POMCP,
    ParticleBelief,
    TigerPOMDP,
    TigerState,
    exact_policy_average,
    solve_exact,
)


class TigerModelTests(unittest.TestCase):
    def setUp(self):
        self.env = TigerPOMDP(noise=0.9, gamma=0.95)
        self.rng = random.Random(1)

    def test_listen_reward_and_state(self):
        state = TigerState(LEFT)
        next_state, reward, done = self.env.transition(state, LISTEN, self.rng)
        self.assertEqual(next_state, state)
        self.assertEqual(reward, -1.0)
        self.assertFalse(done)

    def test_open_rewards(self):
        env = self.env
        rng = self.rng
        s_left = TigerState(LEFT)
        s_right = TigerState(RIGHT)

        _, r, done = env.transition(s_left, OPEN_LEFT, rng)
        self.assertEqual(r, -100.0)
        self.assertTrue(done)

        _, r, _ = env.transition(s_left, OPEN_RIGHT, rng)
        self.assertEqual(r, 10.0)

        _, r, _ = env.transition(s_right, OPEN_LEFT, rng)
        self.assertEqual(r, 10.0)

        _, r, _ = env.transition(s_right, OPEN_RIGHT, rng)
        self.assertEqual(r, -100.0)

    def test_observation_model(self):
        perfect = TigerPOMDP(noise=1.0)
        rng = random.Random(3)
        self.assertEqual(perfect.sample_observation(TigerState(LEFT), LISTEN, rng), LEFT)
        self.assertEqual(perfect.sample_observation(TigerState(RIGHT), LISTEN, rng), RIGHT)
        self.assertEqual(perfect.sample_observation(TigerState(LEFT), OPEN_LEFT, rng), "none")

        backwards = TigerPOMDP(noise=0.0)
        self.assertEqual(backwards.sample_observation(TigerState(LEFT), LISTEN, rng), RIGHT)
        self.assertEqual(backwards.sample_observation(TigerState(RIGHT), LISTEN, rng), LEFT)

    def test_bayes_update(self):
        env = self.env
        self.assertAlmostEqual(env.bayes_update(0.5, LISTEN, LEFT), 0.9, places=10)
        self.assertAlmostEqual(env.bayes_update(0.5, LISTEN, RIGHT), 0.1, places=10)
        self.assertAlmostEqual(env.bayes_update(0.9, LISTEN, LEFT), 0.987804878, places=8)
        # Non-listen actions do not change the belief.
        self.assertEqual(env.bayes_update(0.37, OPEN_LEFT, "none"), 0.37)


class ParticleFilterTests(unittest.TestCase):
    def test_one_observation(self):
        env = TigerPOMDP(noise=0.9)
        rng = random.Random(5)
        belief = ParticleBelief(env, 20000, rng)
        # Start from an (almost) exact 50/50 split.
        belief.particles = [TigerState(LEFT if i % 2 else RIGHT) for i in range(20000)]
        belief.update(LISTEN, LEFT)
        self.assertAlmostEqual(belief.fraction_left(), 0.9, delta=0.015)

    def test_two_observations_sharpen(self):
        env = TigerPOMDP(noise=0.9)
        rng = random.Random(9)
        belief = ParticleBelief(env, 20000, rng)
        belief.particles = [TigerState(LEFT if i % 2 else RIGHT) for i in range(20000)]
        belief.update(LISTEN, RIGHT)
        belief.update(LISTEN, RIGHT)
        self.assertLess(belief.fraction_left(), 0.02)


class ExactSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TigerPOMDP(noise=0.9, gamma=0.95)
        cls.exact = solve_exact(cls.env, points=401)

    def test_value_symmetry(self):
        for b in (0.05, 0.2, 0.5, 0.8, 0.95):
            self.assertAlmostEqual(
                self.exact.value_at(b), self.exact.value_at(1 - b), places=4
            )

    def test_certain_beliefs(self):
        left = self.exact.values_at(0.0)
        right = self.exact.values_at(1.0)
        self.assertEqual(left["best"], OPEN_LEFT)
        self.assertEqual(right["best"], OPEN_RIGHT)
        self.assertAlmostEqual(left["best_value"], 10.0, places=3)
        self.assertAlmostEqual(right["best_value"], 10.0, places=3)

    def test_threshold_and_uniform_value(self):
        self.assertGreater(self.exact.threshold, 0.9)
        self.assertLess(self.exact.threshold, 1.0)
        self.assertGreater(self.exact.value_at(0.5), -10)
        self.assertLess(self.exact.value_at(0.5), 10)
        self.assertGreater(
            self.exact.values_at(0.5)["listen"],
            self.exact.values_at(0.5)["open-left"],
        )
        self.assertEqual(
            self.exact.values_at(self.exact.threshold)["best"], OPEN_RIGHT
        )

    def test_exact_policy_average_reasonable(self):
        avg = exact_policy_average(self.env, self.exact, episodes=400)
        self.assertGreater(avg, 0.0)
        self.assertLess(avg, 10.0)


class PlannerTests(unittest.TestCase):
    def make_planner(self, seed=11, sims=3000):
        env = TigerPOMDP(noise=0.9, gamma=0.95)
        rng = random.Random(seed)
        planner = POMCP(env, simulations=sims, exploration=100.0, rng=rng)
        belief = ParticleBelief(env, 1000, rng)
        return env, rng, planner, belief

    def test_plans_listen_when_uncertain(self):
        env, rng, planner, belief = self.make_planner()
        plan = planner.plan(belief)
        self.assertEqual(plan.action, LISTEN)
        q_listen = plan.q(LISTEN)
        q_open = max(plan.q(OPEN_LEFT), plan.q(OPEN_RIGHT))
        self.assertGreater(q_listen, q_open)

    def test_plans_open_when_certain(self):
        env, rng, planner, belief = self.make_planner()

        belief.particles = [TigerState(LEFT)] * 1000
        planner.reset()
        plan = planner.plan(belief)
        self.assertEqual(plan.action, OPEN_RIGHT)

        belief.particles = [TigerState(RIGHT)] * 1000
        planner.reset()
        plan = planner.plan(belief)
        self.assertEqual(plan.action, OPEN_LEFT)

    def test_tree_reuse_follows_history(self):
        env, rng, planner, belief = self.make_planner(sims=1000)
        planner.plan(belief)
        planner.advance(LISTEN, LEFT)
        plan = planner.plan(belief)
        self.assertIn(plan.action, env.actions)


class EndToEndTests(unittest.TestCase):
    def test_episode_runs_and_belief_tracks(self):
        env = TigerPOMDP(noise=0.9, gamma=0.95)
        exact = solve_exact(env, points=301)
        rng = random.Random(42)
        from pomcp import run_episode

        results = [
            run_episode(env, exact, 1200, 100.0, 800, rng) for _ in range(6)
        ]
        for result in results:
            self.assertGreaterEqual(result.steps, 1)
            self.assertLessEqual(result.steps, 80)
            self.assertIn(result.opened, ("L", "R"))
            self.assertLess(result.belief_mae, 0.08)
            self.assertGreaterEqual(result.decisions, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
