"""Train a PPO agent on the full-quest environment through an explicit CLI.

Importing this module or running it with ``--help`` has no side effects: it
does not train, import Stable-Baselines3/torch/pygame, or create files and
directories. A run needs an explicit, *new* ``--output-dir`` and a bounded
``--timesteps`` budget. All checks (arguments, output directory, the optional
resume checkpoint and its observation/action compatibility) happen before any
output is created. Every artifact of a run is written inside that fresh
directory::

    <output-dir>/run_config.json            settings of this run
    <output-dir>/train.monitor.csv          training episode statistics
    <output-dir>/eval.monitor.csv           evaluation episode statistics
    <output-dir>/evaluations.npz            EvalCallback results (if evaluated)
    <output-dir>/best_model/best_model.zip  best evaluated policy (if evaluated)
    <output-dir>/ppo_full_quest.zip         final checkpoint
    <output-dir>/tensorboard/               only with --tensorboard

The committed checkpoint ``rl/ppo_full_quest.zip`` is never written, and an
existing directory is never reused, so earlier runs are not overwritten.

Training starts from scratch by default. ``--resume PATH`` continues from a
complete Stable-Baselines3 PPO checkpoint, keeping that checkpoint's
hyperparameters. The checkpoint's observation and action spaces must match
``FullQuestEnv``. A missing, unreadable or incompatible checkpoint is an error.
The script never falls back to training from scratch.

Security: SB3 checkpoints contain pickled Python objects, and loading one can
execute arbitrary code. Only resume from checkpoints you trust. This script is
not a sandbox.

Defaults match the original training script: ``MultiInputPolicy``, a 1024-step
``TimeLimit``, learning_rate 1e-4, n_steps 512, batch_size 64, n_epochs 20,
gamma 0.99, gae_lambda 0.95, ent_coef 0.1, and evaluation every 10,240 steps
over 5 deterministic episodes. Smaller ``--n-steps``/``--n-epochs``/
``--eval-freq``/``--eval-episodes`` values are an explicit opt-in for a quick
smoke run. A smoke run only shows that the pipeline runs. It does not produce a
good policy.

Seeds: ``--seed`` seeds the model and the training environment. The
evaluation environment uses ``seed + EVAL_SEED_OFFSET``. SB3 seeds the
process-global ``random``/NumPy/torch generators, so their states are saved
before a run and restored afterwards. Callers' global RNG streams are
unchanged.

Examples (from the repository root; ``rl/logs/`` is git-ignored):
    python rl/train_full_quest.py --output-dir rl/logs/run-001 --timesteps 20000000
    python rl/train_full_quest.py --output-dir rl/logs/smoke-001 --timesteps 128 \\
        --n-steps 64 --n-epochs 1 --eval-freq 64 --eval-episodes 1 --device cpu
    python rl/train_full_quest.py --output-dir rl/logs/resume-001 \\
        --timesteps 1000000 --resume /path/to/trusted_checkpoint.zip
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMMITTED_CHECKPOINT = REPO_ROOT / "rl" / "ppo_full_quest.zip"
CHECKPOINT_NAME = "ppo_full_quest.zip"
RUN_CONFIG_NAME = "run_config.json"

POLICY = "MultiInputPolicy"
MAX_EPISODE_STEPS = 1024
BATCH_SIZE = 64
FIXED_HYPERPARAMETERS: dict[str, float | int] = {
    "learning_rate": 1e-4,
    "batch_size": BATCH_SIZE,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.1,
}
DEFAULT_N_STEPS = 512
DEFAULT_N_EPOCHS = 20
DEFAULT_EVAL_FREQ = 10_240  # 1024 * 10
DEFAULT_EVAL_EPISODES = 5
DEFAULT_SEED = 0
EVAL_SEED_OFFSET = 10_000

MAX_TIMESTEPS = 20_000_000  # The original script's full training budget.
MAX_N_STEPS = 1_000_000
MAX_N_EPOCHS = 100
MAX_EVAL_EPISODES = 1_000
MAX_SEED = 2**32 - 1  # Upper bound accepted by numpy.random.seed.
DEVICES = ("auto", "cpu", "cuda", "mps")


class TrainingError(RuntimeError):
    """Raised for an invalid output directory or a bad resume checkpoint."""


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Validated settings for one training run.

    Attributes:
        output_dir: Fresh directory that receives every artifact of the run.
        timesteps: Training budget. When resuming, this is the number of
            additional timesteps.
        resume: Trusted full SB3 PPO checkpoint to continue from, or ``None``
            to train from scratch.
        seed: Seed for the model and the training environment.
        n_steps: Rollout steps per update, or ``None`` when resuming (the
            checkpoint's value is kept).
        n_epochs: Optimisation epochs per update, or ``None`` when resuming.
        eval_freq: Evaluate every this many training steps.
        eval_episodes: Deterministic episodes per evaluation.
        device: Torch device passed to SB3.
        tensorboard: Whether to write TensorBoard logs into the output dir.
    """

    output_dir: Path
    timesteps: int
    resume: Path | None = None
    seed: int = DEFAULT_SEED
    n_steps: int | None = DEFAULT_N_STEPS
    n_epochs: int | None = DEFAULT_N_EPOCHS
    eval_freq: int = DEFAULT_EVAL_FREQ
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    device: str = "auto"
    tensorboard: bool = False


# --------------------------------------------------------------------------- #
# Lazy dependencies (keep import and --help free of heavy imports)
# --------------------------------------------------------------------------- #
def _default_ppo_cls() -> Any:
    """Imports and returns the Stable-Baselines3 ``PPO`` class.

    Returns:
        The ``stable_baselines3.PPO`` class.
    """
    from stable_baselines3 import PPO

    return PPO


def _default_env_factory() -> Callable[[], Any]:
    """Imports and returns the ``FullQuestEnv`` class.

    Returns:
        A zero-argument callable that creates a headless ``FullQuestEnv``.
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from rl.environments.full_quest_env_pass import FullQuestEnv

    return FullQuestEnv


@contextlib.contextmanager
def isolated_global_rng() -> Iterator[None]:
    """Restores process-global RNG states after the enclosed block.

    SB3 seeding calls ``random.seed``, ``numpy.random.seed`` and
    ``torch.manual_seed``. This context saves the Python, NumPy and torch
    (CPU, plus CUDA/MPS when available) generator states and restores them on
    exit, including when an exception is raised.

    Yields:
        None.
    """
    import numpy as np
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = None
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        mps_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


# --------------------------------------------------------------------------- #
# Validation helpers (no side effects)
# --------------------------------------------------------------------------- #
def check_output_dir(path: Path | str) -> Path:
    """Checks that ``path`` can be created as a fresh run directory.

    Nothing is created here. The directory must not exist yet, including as a
    file or a dangling symlink. Its nearest existing ancestor must be a
    writable directory.

    Args:
        path: Requested output directory. A relative path is resolved against
            the current working directory.

    Returns:
        The absolute output directory path.

    Raises:
        TrainingError: If the path exists or cannot be created.
    """
    path = Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise TrainingError(
            f"output directory already exists: {path}; choose a new path "
            "(existing runs and files are never reused or overwritten)"
        )
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise TrainingError(
            f"cannot create output directory {path}: {ancestor} is not a directory"
        )
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise TrainingError(
            f"cannot create output directory {path}: {ancestor} is not writable"
        )
    return path


def load_checkpoint(path: Path | str, *, device: str, ppo_cls: Any) -> Any:
    """Loads a complete, trusted SB3 PPO checkpoint from a local file.

    The whole checkpoint is loaded with ``PPO.load`` (architecture,
    hyperparameters and weights). Individual tensors are never copied out of
    it. ``PPO.load`` unpickles data, so ``path`` must be trusted.

    Args:
        path: Local checkpoint path. Nothing is downloaded.
        device: Torch device for the loaded model.
        ppo_cls: Class providing ``load`` (normally ``stable_baselines3.PPO``).

    Returns:
        The loaded model.

    Raises:
        TrainingError: If the file is missing or cannot be loaded.
    """
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise TrainingError(f"resume checkpoint not found: {path}")
    try:
        return ppo_cls.load(str(path), device=device)
    except Exception as exc:
        raise TrainingError(
            f"could not load PPO checkpoint {path}: {type(exc).__name__}: {exc}"
        ) from exc


def check_compatibility(model: Any, env: Any) -> None:
    """Requires identical observation and action spaces.

    Args:
        model: Loaded model exposing ``observation_space``/``action_space``.
        env: Environment to train on.

    Raises:
        TrainingError: If either space differs.
    """
    model_obs = getattr(model, "observation_space", None)
    model_act = getattr(model, "action_space", None)
    if model_obs != env.observation_space:
        raise TrainingError(
            "resume checkpoint observation space does not match FullQuestEnv: "
            f"{model_obs} != {env.observation_space}"
        )
    if model_act != env.action_space:
        raise TrainingError(
            "resume checkpoint action space does not match FullQuestEnv: "
            f"{model_act} != {env.action_space}"
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_env(env_factory: Callable[[], Any], monitor_path: Path) -> Any:
    """Creates a ``Monitor(TimeLimit(env))`` with the training horizon.

    Args:
        env_factory: Zero-argument callable returning a Gymnasium env.
        monitor_path: Monitor file prefix (``.monitor.csv`` is appended).

    Returns:
        The wrapped environment.
    """
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.monitor import Monitor

    env = env_factory()
    try:
        return Monitor(
            TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS), str(monitor_path)
        )
    except BaseException:
        env.close()
        raise


def _describe(obj: Any) -> str:
    """Returns ``module.qualname`` for a class or callable, else its repr."""
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None)
    return f"{module}.{name}" if module and name else repr(obj)


def _write_run_config(
    output_dir: Path, config: TrainingConfig, eval_seed: int, env_factory: Any
) -> None:
    """Writes the settings of this run to a new ``run_config.json``.

    Args:
        output_dir: Fresh run directory.
        config: Run settings.
        eval_seed: Seed of the evaluation environment.
        env_factory: Environment factory, recorded by name.
    """
    if config.resume is None:
        hyperparameters: Any = {
            **FIXED_HYPERPARAMETERS,
            "n_steps": config.n_steps,
            "n_epochs": config.n_epochs,
        }
    else:
        hyperparameters = "kept from the resume checkpoint"
    payload = {
        "algorithm": "PPO",
        "policy": POLICY,
        "environment": _describe(env_factory),
        "max_episode_steps": MAX_EPISODE_STEPS,
        "timesteps": config.timesteps,
        "resume": None if config.resume is None else str(config.resume),
        "seed": config.seed,
        "eval_seed": eval_seed,
        "eval_freq": config.eval_freq,
        "eval_episodes": config.eval_episodes,
        "device": config.device,
        "hyperparameters": hyperparameters,
        "note": (
            "Run settings only. A short smoke run does not produce a trained "
            "policy; evaluate checkpoints with rl/run_full_quest.py --model."
        ),
    }
    # Mode "x" refuses to replace an existing file.
    with open(output_dir / RUN_CONFIG_NAME, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def train(
    config: TrainingConfig,
    *,
    ppo_cls: Any = None,
    env_factory: Callable[[], Any] | None = None,
) -> Path:
    """Validates everything, then trains and saves into a fresh directory.

    Args:
        config: Run settings.
        ppo_cls: PPO class (constructor and ``load``). Defaults to
            ``stable_baselines3.PPO``. Tests inject fakes here.
        env_factory: Zero-argument environment factory. Defaults to
            ``FullQuestEnv``.

    Returns:
        Path of the final checkpoint inside ``config.output_dir``.

    Raises:
        TrainingError: On any validation failure. Nothing has been created
            on disk in that case.
    """
    if ppo_cls is None:
        ppo_cls = _default_ppo_cls()
    if env_factory is None:
        env_factory = _default_env_factory()

    with isolated_global_rng():
        # ---- validation: nothing is written before this block completes ----
        output_dir = check_output_dir(config.output_dir)
        if config.tensorboard and importlib.util.find_spec("tensorboard") is None:
            raise TrainingError("--tensorboard requires the 'tensorboard' package")
        resumed = None
        if config.resume is not None:
            resumed = load_checkpoint(
                config.resume, device=config.device, ppo_cls=ppo_cls
            )
            reference = env_factory()
            try:
                check_compatibility(resumed, reference)
            finally:
                reference.close()

        # ---- outputs: a single fresh directory ----
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise TrainingError(
                f"could not create output directory {output_dir}: {exc}"
            ) from exc
        return _run(config, output_dir, resumed, ppo_cls, env_factory)


def _run(
    config: TrainingConfig,
    output_dir: Path,
    resumed: Any,
    ppo_cls: Any,
    env_factory: Callable[[], Any],
) -> Path:
    """Builds environments, trains, saves and always closes the environments.

    Args:
        config: Run settings.
        output_dir: Freshly created run directory.
        resumed: Loaded, compatibility-checked model, or ``None``.
        ppo_cls: PPO class used when training from scratch.
        env_factory: Zero-argument environment factory.

    Returns:
        Path of the final checkpoint.

    Raises:
        TrainingError: If the final checkpoint path already exists.
    """
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    tensorboard_log = str(output_dir / "tensorboard") if config.tensorboard else None
    eval_seed = config.seed + EVAL_SEED_OFFSET
    _write_run_config(output_dir, config, eval_seed, env_factory)

    train_env = eval_env = None
    try:
        train_env = DummyVecEnv([lambda: make_env(env_factory, output_dir / "train")])
        eval_env = DummyVecEnv([lambda: make_env(env_factory, output_dir / "eval")])
        # Seeds the next reset only; later resets continue the env's own stream.
        eval_env.seed(eval_seed)

        if resumed is None:
            model = ppo_cls(
                POLICY,
                train_env,
                n_steps=config.n_steps,
                n_epochs=config.n_epochs,
                seed=config.seed,
                device=config.device,
                tensorboard_log=tensorboard_log,
                verbose=1,
                **FIXED_HYPERPARAMETERS,
            )
            reset_num_timesteps = True
        else:
            model = resumed
            model.set_env(train_env)  # SB3 re-checks the spaces here as well.
            model.set_random_seed(config.seed)
            # Never log to a path stored in the checkpoint (it may point outside
            # the chosen output directory).
            model.tensorboard_log = tensorboard_log
            reset_num_timesteps = False

        callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir / "best_model"),
            log_path=str(output_dir),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        model.learn(
            total_timesteps=config.timesteps,
            callback=callback,
            progress_bar=False,  # No tqdm/rich requirement.
            reset_num_timesteps=reset_num_timesteps,
        )

        final_path = output_dir / CHECKPOINT_NAME
        if final_path.exists():
            raise TrainingError(f"refusing to overwrite existing file {final_path}")
        model.save(str(final_path))
    finally:
        for env in (eval_env, train_env):
            if env is not None:
                env.close()
    return final_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _bounded_int(name: str, minimum: int, maximum: int) -> Callable[[str], int]:
    """Builds an argparse ``type`` that accepts integers in a closed range.

    Args:
        name: Option name used in error messages.
        minimum: Smallest accepted value.
        maximum: Largest accepted value.

    Returns:
        A parser function that raises ``argparse.ArgumentTypeError`` on bad
        input.
    """

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


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO on FullQuestEnv into a NEW output directory. Trains from "
            "scratch unless --resume is given. Defaults match the original "
            "training settings."
        ),
        epilog=(
            "Smoke run (checks the pipeline only; not a trained policy): "
            "--timesteps 128 --n-steps 64 --n-epochs 1 --eval-freq 64 "
            "--eval-episodes 1 --device cpu. "
            "--resume loads a full SB3 checkpoint, which unpickles data and can "
            "execute code: use only TRUSTED files (this is not a sandbox)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "new directory for all run outputs; must not exist "
            "(example: rl/logs/run-001, git-ignored)"
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=_bounded_int("timesteps", 1, MAX_TIMESTEPS),
        required=True,
        help=(
            f"training timesteps in [1, {MAX_TIMESTEPS}] (additional steps when "
            "resuming; SB3 rounds up to whole rollouts)"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "TRUSTED full SB3 PPO checkpoint (.zip) to continue from; must match "
            "FullQuestEnv spaces; no fallback to scratch"
        ),
    )
    parser.add_argument(
        "--seed",
        type=_bounded_int("seed", 0, MAX_SEED),
        default=DEFAULT_SEED,
        help=(
            f"training seed (default {DEFAULT_SEED}); evaluation uses "
            f"seed+{EVAL_SEED_OFFSET}"
        ),
    )
    parser.add_argument(
        "--n-steps",
        type=_bounded_int("n-steps", BATCH_SIZE, MAX_N_STEPS),
        default=None,
        help=(
            f"rollout steps per update (default {DEFAULT_N_STEPS}; minimum "
            f"{BATCH_SIZE} = batch size; not allowed with --resume)"
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=_bounded_int("n-epochs", 1, MAX_N_EPOCHS),
        default=None,
        help=f"epochs per update (default {DEFAULT_N_EPOCHS}; not allowed with --resume)",
    )
    parser.add_argument(
        "--eval-freq",
        type=_bounded_int("eval-freq", 1, MAX_TIMESTEPS),
        default=DEFAULT_EVAL_FREQ,
        help=f"evaluate every N training steps (default {DEFAULT_EVAL_FREQ})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=_bounded_int("eval-episodes", 1, MAX_EVAL_EPISODES),
        default=DEFAULT_EVAL_EPISODES,
        help=f"deterministic episodes per evaluation (default {DEFAULT_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="torch device (default auto; use cpu for a smoke run)",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="also write TensorBoard logs to <output-dir>/tensorboard",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    """Parses command-line arguments into a ``TrainingConfig``.

    Args:
        argv: Argument list without the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        The validated configuration.

    Raises:
        SystemExit: With code 2 on invalid arguments, or 0 for ``--help``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None:
        if args.n_steps is not None or args.n_epochs is not None:
            parser.error(
                "--n-steps/--n-epochs cannot be combined with --resume; a resumed "
                "run keeps the checkpoint's rollout settings"
            )
        n_steps = n_epochs = None
    else:
        n_steps = DEFAULT_N_STEPS if args.n_steps is None else args.n_steps
        n_epochs = DEFAULT_N_EPOCHS if args.n_epochs is None else args.n_epochs
    return TrainingConfig(
        output_dir=args.output_dir,
        timesteps=args.timesteps,
        resume=args.resume,
        seed=args.seed,
        n_steps=n_steps,
        n_epochs=n_epochs,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=args.device,
        tensorboard=args.tensorboard,
    )


def main(
    argv: list[str] | None = None,
    *,
    ppo_cls: Any = None,
    env_factory: Callable[[], Any] | None = None,
) -> int:
    """Runs the command-line interface.

    Args:
        argv: Argument list without the program name.
        ppo_cls: Optional PPO class override (for tests).
        env_factory: Optional environment factory override (for tests).

    Returns:
        0 on success, 1 on a validation error (nothing created).
    """
    config = parse_args(argv)
    try:
        final_path = train(config, ppo_cls=ppo_cls, env_factory=env_factory)
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Training finished; checkpoint saved to {final_path}")
    print(
        "Note: finishing a run does not show the policy is any good. Evaluate it "
        f"with: python rl/run_full_quest.py --model {final_path} --episodes 20 --seed 0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
