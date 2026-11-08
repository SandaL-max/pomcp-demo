"""POMCP on the Tiger POMDP — a small, dependency-free reference implementation.

This mirrors the JavaScript demo in ../js/.  It is deliberately written with
plain Python (no NumPy) so the algorithm is easy to read and run anywhere.

    python3 pomcp.py                       # 20 episodes, 5,000 simulations/step
    python3 pomcp.py --episodes 5 --trace  # step-by-step decisions
    python3 pomcp.py --sweep               # compare simulation budgets

The planner follows Silver & Veness (2010), "Monte-Carlo Planning in Large
POMDPs".  At every decision it builds a tree over action-observation histories,
using UCT and random rollouts, with a particle filter for the belief.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Tiger POMDP
# ---------------------------------------------------------------------------

LEFT = "L"
RIGHT = "R"
LISTEN = "listen"
OPEN_LEFT = "open-left"
OPEN_RIGHT = "open-right"


@dataclass(frozen=True)
class TigerState:
    tiger: str  # "L" or "R"
    done: bool = False


class TigerPOMDP:
    """Episodic Tiger: opening a door ends the episode.

    listen      : -1, hears the tiger's side correctly with probability `noise`
    open-left   : +10 if the tiger is right, -100 if it is left
    open-right  : +10 if the tiger is left, -100 if it is right
    """

    actions: Tuple[str, str, str] = (LISTEN, OPEN_LEFT, OPEN_RIGHT)

    def __init__(
        self,
        noise: float = 0.9,
        gamma: float = 0.95,
        r_correct: float = 10.0,
        r_wrong: float = -100.0,
        listen_cost: float = -1.0,
    ) -> None:
        self.noise = float(noise)
        self.gamma = float(gamma)
        self.r_correct = float(r_correct)
        self.r_wrong = float(r_wrong)
        self.listen_cost = float(listen_cost)

    def initial_state(self, rng: random.Random) -> TigerState:
        return TigerState(LEFT if rng.random() < 0.5 else RIGHT)

    def transition(
        self, state: TigerState, action: str, rng: random.Random
    ) -> Tuple[TigerState, float, bool]:
        if state.done:
            return state, 0.0, True
        if action == LISTEN:
            return state, self.listen_cost, False

        opened_left = action == OPEN_LEFT
        correct = (opened_left and state.tiger == RIGHT) or (
            not opened_left and state.tiger == LEFT
        )
        return (
            TigerState(state.tiger, done=True),
            self.r_correct if correct else self.r_wrong,
            True,
        )

    def sample_observation(
        self, state: TigerState, action: str, rng: random.Random
    ) -> str:
        if action != LISTEN or state.done:
            return "none"
        correct = rng.random() < self.noise
        heard_left = (state.tiger == LEFT) == correct
        return LEFT if heard_left else RIGHT

    def observation_probability(
        self, state: TigerState, action: str, observation: str
    ) -> float:
        if action != LISTEN or state.done:
            return 1.0 if observation == "none" else 0.0
        heard_left = observation == LEFT
        correct = (heard_left and state.tiger == LEFT) or (
            not heard_left and state.tiger == RIGHT
        )
        return self.noise if correct else 1.0 - self.noise

    def step(
        self, state: TigerState, action: str, rng: random.Random
    ) -> Tuple[TigerState, str, float, bool]:
        next_state, reward, done = self.transition(state, action, rng)
        observation = self.sample_observation(next_state, action, rng)
        return next_state, observation, reward, done

    def is_terminal(self, state: TigerState) -> bool:
        return state.done

    def bayes_update(self, b: float, action: str, observation: str) -> float:
        """Exact posterior P(tiger left) after one listen observation."""
        if action != LISTEN:
            return b
        p = self.noise
        like_left = p if observation == LEFT else 1.0 - p
        like_right = (1.0 - p) if observation == LEFT else p
        numerator = b * like_left
        denominator = numerator + (1.0 - b) * like_right
        return numerator / denominator if denominator > 1e-15 else b


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------


class ParticleBelief:
    """Sequential importance resampling over sampled states."""

    def __init__(self, env: TigerPOMDP, count: int, rng: random.Random) -> None:
        self.env = env
        self.rng = rng
        self.particles: List[TigerState] = [
            env.initial_state(rng) for _ in range(count)
        ]

    def __len__(self) -> int:
        return len(self.particles)

    def sample(self, rng: Optional[random.Random] = None) -> TigerState:
        return (rng or self.rng).choice(self.particles)

    def fraction_left(self) -> float:
        return sum(s.tiger == LEFT for s in self.particles) / len(self.particles)

    def update(self, action: str, observation: str) -> None:
        env, rng = self.env, self.rng
        states: List[TigerState] = []
        weights: List[float] = []
        max_weight = 0.0

        for particle in self.particles:
            state, _, _ = env.transition(particle, action, rng)
            weight = env.observation_probability(state, action, observation)
            states.append(state)
            weights.append(max(0.0, weight))
            max_weight = max(max_weight, weight)

        n = len(states)
        if max_weight <= 0.0:  # impoverished filter: propagate unweighted
            self.particles = states
            return

        weights = [w / max_weight for w in weights]
        # Randomise the order before systematic resampling: an unlucky fixed
        # ordering (e.g. all high-weight particles first) can make systematic
        # sampling systematically skip a low-weight group.
        order = list(range(n))
        rng.shuffle(order)
        states = [states[i] for i in order]
        weights = [weights[i] for i in order]

        total = sum(weights)
        step = total / n
        u = rng.random() * step
        acc = 0.0
        j = 0
        resampled: List[TigerState] = []
        for _ in range(n):
            while j < n - 1 and u > acc + weights[j]:
                acc += weights[j]
                j += 1
            resampled.append(states[j])
            u += step
        self.particles = resampled


# ---------------------------------------------------------------------------
# POMCP
# ---------------------------------------------------------------------------


class Node:
    __slots__ = ("visited", "N", "Na", "Qa", "children")

    def __init__(self) -> None:
        self.visited = False
        self.N = 0
        self.Na: Dict[str, int] = {}
        self.Qa: Dict[str, float] = {}
        self.children: Dict[str, "Node"] = {}


@dataclass
class Plan:
    action: str
    rows: List[dict]
    simulations: int
    root_n: int
    nodes: int
    depth: int

    def q(self, action: str) -> Optional[float]:
        for row in self.rows:
            if row["action"] == action:
                return row["Q"]
        return None


class POMCP:
    def __init__(
        self,
        env: TigerPOMDP,
        simulations: int = 5000,
        exploration: float = 100.0,
        gamma: Optional[float] = None,
        max_depth: int = 40,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.env = env
        self.simulations = int(simulations)
        self.exploration = float(exploration)
        self.gamma = env.gamma if gamma is None else float(gamma)
        self.max_depth = int(max_depth)
        self.rng = rng or random.Random(0)
        self.root = Node()
        self.total_simulations = 0

    def reset(self) -> None:
        self.root = Node()

    def advance(self, action: str, observation: str) -> None:
        self.root = self.root.children.get(f"{action}|{observation}", Node())

    # -- tree search --------------------------------------------------------

    def _select(self, node: Node) -> str:
        actions = self.env.actions
        best = actions[0]
        best_score = -math.inf
        log_n = math.log(max(1, node.N))
        for action in actions:
            n_a = node.Na.get(action, 0)
            if n_a == 0:
                return action
            score = node.Qa[action] + self.exploration * math.sqrt(log_n / n_a)
            if score > best_score:
                best_score = score
                best = action
        return best

    def _rollout(self, state: TigerState, depth: int) -> float:
        total = 0.0
        discount = 1.0
        for _ in range(depth, self.max_depth):
            if self.env.is_terminal(state):
                break
            action = self.rng.choice(self.env.actions)
            state, _, reward, done = self.env.step(state, action, self.rng)
            total += discount * reward
            discount *= self.gamma
            if done:
                break
        return total

    def _simulate(self, state: TigerState, node: Node, depth: int) -> float:
        if depth >= self.max_depth or self.env.is_terminal(state):
            return 0.0

        if not node.visited:
            node.visited = True
            return self._rollout(state, depth)

        action = self._select(node)
        next_state, reward, done = self.env.transition(state, action, self.rng)
        observation = self.env.sample_observation(next_state, action, self.rng)

        if done:
            ret = reward
        else:
            key = f"{action}|{observation}"
            child = node.children.get(key)
            if child is None:
                child = Node()
                node.children[key] = child
            ret = reward + self.gamma * self._simulate(next_state, child, depth + 1)

        node.N += 1
        n_a = node.Na.get(action, 0) + 1
        node.Na[action] = n_a
        old_q = node.Qa.get(action, 0.0)
        node.Qa[action] = old_q + (ret - old_q) / n_a
        return ret

    def _count_nodes(self, node: Node) -> int:
        return 1 + sum(self._count_nodes(child) for child in node.children.values())

    def _tree_depth(self, node: Node, depth: int = 0) -> int:
        if not node.children:
            return depth
        return max(self._tree_depth(child, depth + 1) for child in node.children.values())

    # -- public -------------------------------------------------------------

    def plan(self, belief: ParticleBelief, simulations: Optional[int] = None) -> Plan:
        n_simulations = self.simulations if simulations is None else int(simulations)
        for _ in range(n_simulations):
            state = belief.sample(self.rng)
            if not self.env.is_terminal(state):
                self._simulate(state, self.root, 0)
        self.total_simulations += n_simulations

        rows = []
        best_action = None
        best_q = -math.inf
        best_n = -1
        for action in self.env.actions:
            n_a = self.root.Na.get(action, 0)
            q = self.root.Qa.get(action)
            rows.append({"action": action, "N": n_a, "Q": q})
            if q is None:
                continue
            if q > best_q + 1e-9 or (abs(q - best_q) <= 1e-9 and n_a > best_n):
                best_q, best_n, best_action = q, n_a, action
        if best_action is None:
            best_action = self.env.actions[0]

        return Plan(
            action=best_action,
            rows=rows,
            simulations=n_simulations,
            root_n=self.root.N,
            nodes=self._count_nodes(self.root),
            depth=self._tree_depth(self.root),
        )


# ---------------------------------------------------------------------------
# Exact belief-state benchmark (value iteration on a belief grid)
# ---------------------------------------------------------------------------


@dataclass
class ExactSolution:
    env: TigerPOMDP
    V: List[float]
    points: int
    iterations: int
    threshold: float
    _last: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._last = self.points - 1

    def interp(self, b: float) -> float:
        b = min(1.0, max(0.0, b))
        x = b * self._last
        i = int(math.floor(x))
        if i >= self._last:
            return self.V[self._last]
        f = x - i
        return self.V[i] * (1.0 - f) + self.V[i + 1] * f

    def value_at(self, b: float) -> float:
        return self.interp(b)

    def _update(self, b: float, observation: str) -> float:
        p = self.env.noise
        like_left = p if observation == LEFT else 1.0 - p
        like_right = (1.0 - p) if observation == LEFT else p
        numerator = b * like_left
        denominator = numerator + (1.0 - b) * like_right
        return numerator / denominator if denominator > 1e-15 else b

    def values_at(self, b: float) -> Dict[str, float]:
        b = min(1.0, max(0.0, b))
        env = self.env
        open_left = env.r_wrong * b + env.r_correct * (1.0 - b)
        open_right = env.r_correct * b + env.r_wrong * (1.0 - b)
        p_left = env.noise * b + (1.0 - env.noise) * (1.0 - b)
        listen = env.listen_cost + env.gamma * (
            p_left * self.interp(self._update(b, LEFT))
            + (1.0 - p_left) * self.interp(self._update(b, RIGHT))
        )
        best = LISTEN
        best_value = listen
        if open_left > best_value:
            best, best_value = OPEN_LEFT, open_left
        if open_right > best_value:
            best, best_value = OPEN_RIGHT, open_right
        return {
            LISTEN: listen,
            OPEN_LEFT: open_left,
            OPEN_RIGHT: open_right,
            "best": best,
            "best_value": best_value,
            "value": self.interp(b),
        }

    def curve(self, samples: int = 201) -> List[Tuple[float, float]]:
        return [(i / (samples - 1), self.interp(i / (samples - 1))) for i in range(samples)]


def solve_exact(
    env: TigerPOMDP,
    points: int = 501,
    max_iter: int = 4000,
    tol: float = 1e-11,
) -> ExactSolution:
    """Value iteration over a uniform grid on b = P(tiger left)."""
    n = points
    last = n - 1
    V = [0.0] * n
    iterations = 0

    def interp(arr: List[float], b: float) -> float:
        b = min(1.0, max(0.0, b))
        x = b * last
        i = int(math.floor(x))
        if i >= last:
            return arr[last]
        f = x - i
        return arr[i] * (1.0 - f) + arr[i + 1] * f

    def update(b: float, observation: str) -> float:
        p = env.noise
        like_left = p if observation == LEFT else 1.0 - p
        like_right = (1.0 - p) if observation == LEFT else p
        numerator = b * like_left
        denominator = numerator + (1.0 - b) * like_right
        return numerator / denominator if denominator > 1e-15 else b

    for iteration in range(1, max_iter + 1):
        next_v = [0.0] * n
        diff = 0.0
        for i in range(n):
            b = i / last
            open_left = env.r_wrong * b + env.r_correct * (1.0 - b)
            open_right = env.r_correct * b + env.r_wrong * (1.0 - b)
            p_left = env.noise * b + (1.0 - env.noise) * (1.0 - b)
            listen = env.listen_cost + env.gamma * (
                p_left * interp(V, update(b, LEFT))
                + (1.0 - p_left) * interp(V, update(b, RIGHT))
            )
            value = max(open_left, open_right, listen)
            next_v[i] = value
            diff = max(diff, abs(value - V[i]))
        V = next_v
        iterations = iteration
        if diff < tol:
            break

    solution = ExactSolution(env=env, V=V, points=n, iterations=iterations, threshold=1.0)

    # Earliest b >= 0.5 at which opening the right door is optimal.
    scan = 4000
    for i in range(scan + 1):
        b = 0.5 + 0.5 * i / scan
        if solution.values_at(b)["best"] == OPEN_RIGHT:
            solution.threshold = b
            break
    return solution


def exact_policy_average(
    env: TigerPOMDP, exact: ExactSolution, episodes: int = 3000, seed: int = 0xC0FFEE
) -> float:
    """Simulated mean *undiscounted* episode return of the exact threshold policy."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(episodes):
        state = env.initial_state(rng)
        b = 0.5
        ret = 0.0
        for _ in range(80):
            if b >= exact.threshold:
                action = OPEN_RIGHT
            elif b <= 1.0 - exact.threshold:
                action = OPEN_LEFT
            else:
                action = LISTEN
            state, observation, reward, done = env.step(state, action, rng)
            ret += reward
            if done:
                break
            b = env.bayes_update(b, action, observation)
        total += ret
    return total / episodes


# ---------------------------------------------------------------------------
# Evaluation / CLI
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    return_: float
    steps: int
    listens: int
    safe_open: bool
    tiger: str
    opened: str
    action_string: str
    belief_mae: float
    agreements: int
    decisions: int


def run_episode(
    env: TigerPOMDP,
    exact: ExactSolution,
    simulations: int,
    exploration: float,
    particles: int,
    rng: random.Random,
    max_steps: int = 80,
    trace: bool = False,
) -> EpisodeResult:
    planner = POMCP(
        env,
        simulations=simulations,
        exploration=exploration,
        rng=rng,
    )
    belief = ParticleBelief(env, particles, rng)
    state = env.initial_state(rng)
    b = 0.5
    total = 0.0
    steps = 0
    listens = 0
    action_string = ""
    belief_error = 0.0
    belief_updates = 0
    agreements = 0
    decisions = 0

    while steps < max_steps:
        plan = planner.plan(belief)
        exact_values = exact.values_at(b)
        decisions += 1
        if plan.action == exact_values["best"]:
            agreements += 1
        action_string += {"listen": "L", OPEN_LEFT: "l", OPEN_RIGHT: "r"}[plan.action]

        if trace:
            qs = "  ".join(
                f"{row['action']}={row['Q']:+.1f}" if row["Q"] is not None else f"{row['action']}=--"
                for row in plan.rows
            )
            print(
                f"  step {steps + 1:2d}  b={b:.3f}  action={plan.action:<10s} "
                f"[{qs}]"
            )

        state, observation, reward, done = env.step(state, plan.action, rng)
        total += reward
        steps += 1
        if plan.action == LISTEN:
            listens += 1
        if trace:
            obs = observation if plan.action == LISTEN else "-"
            print(f"           reward={reward:+.0f}  obs={obs}")

        if done:
            opened = "L" if plan.action == OPEN_LEFT else "R"
            return EpisodeResult(
                return_=total,
                steps=steps,
                listens=listens,
                safe_open=reward > 0,
                tiger=state.tiger,
                opened=opened,
                action_string=action_string,
                belief_mae=belief_error / belief_updates if belief_updates else 0.0,
                agreements=agreements,
                decisions=decisions,
            )

        belief.update(plan.action, observation)
        b = env.bayes_update(b, plan.action, observation)
        planner.advance(plan.action, observation)
        belief_error += abs(belief.fraction_left() - b)
        belief_updates += 1

    raise RuntimeError("episode did not terminate")


def evaluate(
    env: TigerPOMDP,
    exact: ExactSolution,
    episodes: int,
    simulations: int,
    exploration: float,
    particles: int,
    seed: int,
    trace: bool = False,
) -> List[EpisodeResult]:
    rng = random.Random(seed)
    results: List[EpisodeResult] = []
    for episode in range(episodes):
        if trace:
            print(f"\nEpisode {episode + 1}  (tiger starts hidden)")
        result = run_episode(
            env, exact, simulations, exploration, particles, rng, trace=trace
        )
        results.append(result)
        if trace:
            print(
                f"  -> opened {result.opened}, tiger {result.tiger}, "
                f"return {result.return_:+.0f}, {result.steps} steps"
            )
        elif episodes <= 12:
            print(
                f"  ep {episode + 1:2d}  {result.action_string:<8s} "
                f"return {result.return_:+7.0f}   listens {result.listens}   "
                f"belief MAE {result.belief_mae:.3f}"
            )
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict:
    n = len(results)
    returns = [r.return_ for r in results]
    decisions = sum(r.decisions for r in results)
    agreements = sum(r.agreements for r in results)
    return {
        "episodes": n,
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "stdev": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "safe_rate": sum(r.safe_open for r in results) / n if n else 0.0,
        "mean_listens": statistics.fmean([r.listens for r in results]) if n else 0.0,
        "agreement": agreements / decisions if decisions else 0.0,
        "mean_belief_mae": statistics.fmean([r.belief_mae for r in results]) if n else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="POMCP on the episodic Tiger POMDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--exploration", type=float, default=100.0)
    parser.add_argument("--particles", type=int, default=1200)
    parser.add_argument("--noise", type=float, default=0.9, help="P(correct | listen)")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", action="store_true", help="print every decision")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="compare simulation budgets (uses --episodes for each)",
    )
    args = parser.parse_args(argv)

    env = TigerPOMDP(noise=args.noise, gamma=args.gamma)
    exact = solve_exact(env)
    optimal_avg = exact_policy_average(env, exact)

    print("POMCP — online planning under partial observability")
    print("=" * 58)
    print(
        f"Tiger POMDP   listen accuracy={env.noise:.2f}   "
        f"discount={env.gamma:.2f}   particles={args.particles}"
    )
    print(
        f"Exact solver  threshold P(left)={exact.threshold:.4f}   "
        f"V*(uniform)={exact.value_at(0.5):.3f}   "
        f"iterations={exact.iterations}"
    )
    print(f"Exact policy  simulated mean return={optimal_avg:.2f}")
    print()

    if args.sweep:
        print(f"{'simulations':>12} {'mean':>8} {'stdev':>7} {'safe':>7} "
              f"{'listens':>8} {'match':>7} {'belief MAE':>11}")
        print("-" * 66)
        for simulations in (500, 1000, 2000, 5000, 10000):
            results = evaluate(
                env, exact, args.episodes, simulations, args.exploration,
                args.particles, args.seed,
            )
            s = summarize(results)
            print(
                f"{simulations:>12,} {s['mean_return']:>+8.2f} {s['stdev']:>7.2f} "
                f"{100 * s['safe_rate']:>6.1f}% {s['mean_listens']:>8.2f} "
                f"{100 * s['agreement']:>6.1f}% {s['mean_belief_mae']:>11.3f}"
            )
        print("\n'match' = share of decisions where POMCP chose the exact optimal action.")
        return

    print(
        f"Running {args.episodes} episodes with {args.simulations:,} simulations per step..."
    )
    results = evaluate(
        env, exact, args.episodes, args.simulations, args.exploration,
        args.particles, args.seed, trace=args.trace,
    )
    s = summarize(results)
    print()
    print("Summary")
    print("-" * 58)
    print(f"  mean return          {s['mean_return']:+.2f}  (stdev {s['stdev']:.2f})")
    print(f"  exact policy mean    {optimal_avg:+.2f}")
    print(f"  safe opens           {100 * s['safe_rate']:.1f}%")
    print(f"  listens per episode  {s['mean_listens']:.2f}")
    print(f"  exact-action match   {100 * s['agreement']:.1f}%")
    print(f"  particle belief MAE  {s['mean_belief_mae']:.3f}")
    print()
    print("POMCP is an anytime Monte-Carlo approximation: finite simulations make")
    print("the values noisy and occasionally open a door too early.  The exact")
    print("threshold above is the belief at which opening becomes truly optimal.")


if __name__ == "__main__":
    main()
