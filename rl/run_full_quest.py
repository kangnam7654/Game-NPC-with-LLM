"""Evaluate the committed PPO checkpoint on the full-quest environment.

Default mode is headless: no window, no sleeping. Each episode uses a seeded
reset (``base_seed + episode_index``), the policy acts with
``model.predict(obs, deterministic=True)``, and Gymnasium's ``TimeLimit``
bounds the episode (default 1024 steps, the training horizon). A JSON report
is printed to stdout.

Outcomes:
  * ``solved``    -- the env *terminated*: treasure opened and exit reached.
  * ``truncated`` -- the step limit was hit before the quest was solved.

This measures one checkpoint on a handful of seeds; it is not a benchmark.
The NPC "information" in this environment is rule-based (adjacent + interact);
no LLM is called and nothing is trained or downloaded.

Examples (from any directory):
    python rl/run_full_quest.py --episodes 20 --seed 0
    python rl/run_full_quest.py --episodes 3 --render --delay 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep stdout machine-readable (pygame prints a banner on import otherwise).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from gymnasium.wrappers import TimeLimit  # noqa: E402

from rl.environments.full_quest_env_pass import FullQuestEnv  # noqa: E402

DEFAULT_MODEL_PATH = REPO_ROOT / "rl" / "ppo_full_quest.zip"
TRAINING_MAX_STEPS = 1024  # Matches TimeLimit in rl/train_full_quest.py
DEFAULT_EPISODES = 10
DEFAULT_BASE_SEED = 0
MAX_EPISODES = 10_000
MAX_STEPS_LIMIT = 100_000
MAX_SEED = 2**31 - 1
MAX_DELAY = 5.0

MILESTONES = (
    "location_info",  # talked to the location NPC
    "treasure_found",  # first treasure visit after learning its location
    "password_info",  # talked to the password NPC
    "treasure_opened",  # entered the correct password
    "exit_reached",  # reached the exit with the treasure opened
)

ENV_LOGGER_NAME = "rl.environments.full_quest_env_pass"


class EvaluationError(RuntimeError):
    """Missing, unreadable or incompatible checkpoint."""


# --------------------------------------------------------------------------- #
# Model loading / compatibility
# --------------------------------------------------------------------------- #
def load_model(path: Path | str):
    """Load a local PPO checkpoint (no download). Raises EvaluationError."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"checkpoint not found: {path}")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise EvaluationError(
            f"stable-baselines3 is required to load {path}: {exc}"
        ) from exc
    try:
        return PPO.load(str(path), device="cpu")
    except Exception as exc:
        raise EvaluationError(
            f"could not load PPO checkpoint {path}: {type(exc).__name__}: {exc}"
        ) from exc


def check_compatibility(model: Any, env: Any) -> None:
    """Require identical observation/action spaces (same rule as SB3)."""
    model_obs = getattr(model, "observation_space", None)
    model_act = getattr(model, "action_space", None)
    if model_obs != env.observation_space:
        raise EvaluationError(
            "checkpoint observation space does not match FullQuestEnv: "
            f"{model_obs} != {env.observation_space}"
        )
    if model_act != env.action_space:
        raise EvaluationError(
            "checkpoint action space does not match FullQuestEnv: "
            f"{model_act} != {env.action_space}"
        )


# --------------------------------------------------------------------------- #
# Episode execution
# --------------------------------------------------------------------------- #
def _milestone_state(env: Any) -> dict[str, bool]:
    """Read quest progress from the actual (unwrapped) environment state."""
    base = env.unwrapped
    game = base.game
    return {
        "location_info": bool(game.knows_location),
        "treasure_found": bool(base.visited_treasure_first),
        "password_info": bool(game.knows_password),
        "treasure_opened": bool(game.treasure_opened),
        "exit_reached": bool(
            game.treasure_opened and tuple(game.player_pos) == tuple(game.exit_pos)
        ),
    }


def classify(terminated: bool, truncated: bool) -> str:
    """Termination (quest solved) takes precedence over a same-step time limit."""
    if terminated:
        return "solved"
    if truncated:
        return "truncated"
    return "running"


def run_episode(env: Any, model: Any, seed: int, delay: float = 0.0) -> dict:
    """Run one deterministic episode on a TimeLimit-wrapped env."""
    obs, _ = env.reset(seed=seed)
    first_step: dict[str, int | None] = {name: None for name in MILESTONES}
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += float(reward)
        for name, reached in _milestone_state(env).items():
            if reached and first_step[name] is None:
                first_step[name] = steps
        if delay > 0:
            time.sleep(delay)

    return {
        "seed": seed,
        "outcome": classify(bool(terminated), bool(truncated)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "milestone_first_step": first_step,
    }


def _validate(episodes: int, base_seed: int, max_steps: int, delay: float) -> None:
    if not isinstance(episodes, int) or not 1 <= episodes <= MAX_EPISODES:
        raise ValueError(f"episodes must be an integer in [1, {MAX_EPISODES}]")
    if not isinstance(base_seed, int) or not 0 <= base_seed <= MAX_SEED:
        raise ValueError(f"base_seed must be an integer in [0, {MAX_SEED}]")
    if not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS_LIMIT:
        raise ValueError(f"max_steps must be an integer in [1, {MAX_STEPS_LIMIT}]")
    if not (isinstance(delay, (int, float)) and math.isfinite(delay)) or not (
        0 <= delay <= MAX_DELAY
    ):
        raise ValueError(f"delay must be a number in [0, {MAX_DELAY}]")


def evaluate(
    model: Any,
    *,
    episodes: int,
    base_seed: int,
    max_steps: int,
    render: bool = False,
    delay: float = 0.0,
    env_factory: Callable[..., Any] = FullQuestEnv,
) -> dict:
    """Evaluate ``model`` and return a JSON-serialisable result dict."""
    _validate(episodes, base_seed, max_steps, delay)
    if delay > 0 and not render:
        raise ValueError("delay is only allowed together with render")

    env = None
    try:
        env = TimeLimit(
            env_factory(render_mode="human" if render else None),
            max_episode_steps=max_steps,
        )
        check_compatibility(model, env)
        results = []
        for index in range(episodes):
            result = run_episode(env, model, base_seed + index, delay)
            results.append({"episode": index + 1, **result})
    finally:
        if env is not None:
            env.close()

    solved = sum(r["outcome"] == "solved" for r in results)
    return {
        "episodes": episodes,
        "base_seed": base_seed,
        "max_steps": max_steps,
        "deterministic": True,
        "summary": {
            "solved": solved,
            "truncated": episodes - solved,
            "mean_total_reward": round(
                sum(r["total_reward"] for r in results) / episodes, 4
            ),
            "mean_steps": round(sum(r["steps"] for r in results) / episodes, 2),
            "episodes_reaching_milestone": {
                name: sum(r["milestone_first_step"][name] is not None for r in results)
                for name in MILESTONES
            },
        },
        "results": results,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _bounded_int(name: str, minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{name} must be an integer, got {text!r}"
            ) from None
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be in [{minimum}, {maximum}], got {value}"
            )
        return value

    return parse


def _delay(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"delay must be a number, got {text!r}"
        ) from None
    if not math.isfinite(value) or not 0 <= value <= MAX_DELAY:
        raise argparse.ArgumentTypeError(f"delay must be in [0, {MAX_DELAY}]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the committed PPO checkpoint on FullQuestEnv "
            "(headless by default) and print a JSON report."
        )
    )
    parser.add_argument(
        "--episodes",
        type=_bounded_int("episodes", 1, MAX_EPISODES),
        default=DEFAULT_EPISODES,
        help=f"number of episodes (default {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--seed",
        type=_bounded_int("seed", 0, MAX_SEED),
        default=DEFAULT_BASE_SEED,
        help="base seed; episode i uses seed+i (default 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=_bounded_int("max-steps", 1, MAX_STEPS_LIMIT),
        default=TRAINING_MAX_STEPS,
        help=f"TimeLimit horizon per episode (default {TRAINING_MAX_STEPS}, as in training)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="checkpoint path (default: <repo>/rl/ppo_full_quest.zip)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="open a Pygame window (requires a desktop)",
    )
    parser.add_argument(
        "--delay",
        type=_delay,
        default=0.0,
        help="seconds to sleep per step; only with --render (default 0)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delay > 0 and not args.render:
        parser.error("--delay requires --render")
    if args.seed + args.episodes - 1 > MAX_SEED:
        parser.error(f"seed + episodes - 1 must not exceed {MAX_SEED}")
    return args


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The env logs at INFO when a password is entered; keep output concise.
    logging.getLogger(ENV_LOGGER_NAME).setLevel(logging.WARNING)

    try:
        model = load_model(args.model)
        result = evaluate(
            model,
            episodes=args.episodes,
            base_seed=args.seed,
            max_steps=args.max_steps,
            render=args.render,
            delay=args.delay,
        )
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = {
        "checkpoint": _display_path(args.model),
        "environment": "rl.environments.full_quest_env_pass.FullQuestEnv",
        **result,
        "note": (
            "solved = env terminated (treasure opened, exit reached); "
            "truncated = hit max_steps first. Deterministic policy on the listed "
            "seeds only; not a benchmark."
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
