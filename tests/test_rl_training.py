"""Tests for the safe training CLI in ``rl/train_full_quest.py``.

Most tests drive the CLI with a deterministic fake PPO class, so they check
orchestration (validation order, fail-closed resume, output placement and
environment cleanup) without training. The final test performs one tiny real
PPO run on CPU. It only checks that the pipeline works and is seeded; it says
nothing about policy quality. Nothing here downloads, calls an LLM or opens a
window.
"""

import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from rl import train_full_quest as tq  # noqa: E402
from rl.environments.full_quest_env_pass import FullQuestEnv  # noqa: E402

SCRIPT = REPO_ROOT / "rl" / "train_full_quest.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def committed_hash():
    return sha256(tq.COMMITTED_CHECKPOINT)


@pytest.fixture(autouse=True)
def committed_checkpoint_unchanged(committed_hash):
    yield
    assert sha256(tq.COMMITTED_CHECKPOINT) == committed_hash


@pytest.fixture(scope="module")
def env_spaces():
    env = FullQuestEnv()
    try:
        return env.observation_space, env.action_space
    finally:
        env.close()


def argv_for(output_dir, *extra, timesteps="128"):
    return ["--output-dir", str(output_dir), "--timesteps", timesteps, *extra]


def rl_dir_listing():
    return sorted(p.name for p in (REPO_ROOT / "rl").iterdir())


# --------------------------------------------------------------------------- #
# Deterministic test doubles
# --------------------------------------------------------------------------- #
class RecordingEnv(FullQuestEnv):
    """Real FullQuestEnv that records whether ``close`` was called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


def recording_factory():
    created = []

    def factory():
        env = RecordingEnv()
        created.append(env)
        return env

    return factory, created


class _FakeModelMixin:
    """Shared fake behaviour: record calls, perturb global RNG, write files."""

    def _init_records(self):
        self.learn_calls = []
        self.saved = []
        self.seeds = []
        self.set_env_calls = []

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        # Mimic SB3, which touches the process-global generators.
        random.random()
        np.random.rand()
        torch.rand(1)
        if self.learn_error is not None:
            raise self.learn_error
        return self

    def save(self, path):
        self.saved.append(Path(path))
        Path(path).write_bytes(b"fake checkpoint")

    def set_env(self, env):
        self.set_env_calls.append(env)
        self.env = env

    def set_random_seed(self, seed):
        self.seeds.append(seed)


class LoadedFakeModel(_FakeModelMixin):
    """Stand-in for a model returned by ``PPO.load``."""

    def __init__(self, observation_space, action_space, learn_error=None):
        self._init_records()
        self.observation_space = observation_space
        self.action_space = action_space
        self.learn_error = learn_error
        self.env = None
        # A checkpoint may carry a machine-specific log path; it must be ignored.
        self.tensorboard_log = "rl/logs/"


def make_fake_ppo(loaded=None, learn_error=None):
    """Returns a fresh fake PPO class with per-test call records."""

    class FakePPO(_FakeModelMixin):
        created = []
        loads = []

        def __init__(self, policy, env, **kwargs):
            self._init_records()
            self.policy_name = policy
            self.env = env
            self.kwargs = kwargs
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.tensorboard_log = kwargs.get("tensorboard_log")
            self.learn_error = learn_error
            FakePPO.created.append(self)

        @classmethod
        def load(cls, path, device="auto", **kwargs):
            cls.loads.append((path, device))
            if loaded is None:
                raise AssertionError("load was not expected")
            return loaded

    return FakePPO


class ExplodingPPO:
    """Fails loudly if validation lets execution reach model creation."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("PPO must not be constructed")

    @classmethod
    def load(cls, *args, **kwargs):
        raise AssertionError("PPO.load must not be called")


@pytest.fixture
def seeded_global_rng():
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    try:
        yield
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def assert_same_rng(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


# --------------------------------------------------------------------------- #
# Import / --help safety
# --------------------------------------------------------------------------- #
def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    return env


def test_import_has_no_side_effects(tmp_path):
    before = rl_dir_listing()
    code = (
        "import sys, rl.train_full_quest\n"
        "heavy = ('stable_baselines3', 'torch', 'pygame', 'gymnasium',\n"
        "         'rl.environments.full_quest_env_pass')\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert list(tmp_path.iterdir()) == []
    assert rl_dir_listing() == before


def test_help_has_no_side_effects(tmp_path):
    before = rl_dir_listing()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for text in ("--output-dir", "--timesteps", "--resume", "TRUSTED"):
        assert text in result.stdout
    assert list(tmp_path.iterdir()) == []
    assert rl_dir_listing() == before


def test_help_in_process_never_reaches_training(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        tq.main(["--help"], ppo_cls=ExplodingPPO)
    assert excinfo.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make_argv",
    [
        lambda out: ["--timesteps", "128"],  # missing --output-dir
        lambda out: ["--output-dir", str(out)],  # missing --timesteps
        lambda out: argv_for(out, timesteps="0"),
        lambda out: argv_for(out, timesteps="-5"),
        lambda out: argv_for(out, timesteps="1.5"),
        lambda out: argv_for(out, timesteps="lots"),
        lambda out: argv_for(out, timesteps=str(tq.MAX_TIMESTEPS + 1)),
        lambda out: argv_for(out, "--n-steps", str(tq.BATCH_SIZE - 1)),
        lambda out: argv_for(out, "--n-epochs", "0"),
        lambda out: argv_for(out, "--eval-freq", "0"),
        lambda out: argv_for(out, "--eval-episodes", "0"),
        lambda out: argv_for(out, "--seed", "-1"),
        lambda out: argv_for(out, "--device", "tpu"),
        lambda out: argv_for(out, "--resume", "x.zip", "--n-steps", "64"),
        lambda out: argv_for(out, "--resume", "x.zip", "--n-epochs", "1"),
    ],
)
def test_cli_rejects_invalid_arguments(make_argv, tmp_path, capsys):
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as excinfo:
        tq.main(make_argv(out), ppo_cls=ExplodingPPO)
    assert excinfo.value.code == 2
    assert not out.exists()


def test_defaults_match_original_settings(tmp_path):
    config = tq.parse_args(argv_for(tmp_path / "run"))
    assert config.resume is None
    assert config.seed == 0
    assert config.n_steps == 512 and config.n_epochs == 20
    assert config.eval_freq == 10_240 and config.eval_episodes == 5
    assert config.device == "auto" and config.tensorboard is False
    assert tq.MAX_EPISODE_STEPS == 1024
    assert tq.POLICY == "MultiInputPolicy"
    assert tq.FIXED_HYPERPARAMETERS == {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.1,
    }


# --------------------------------------------------------------------------- #
# Orchestration with a fake PPO
# --------------------------------------------------------------------------- #
def test_scratch_run_uses_original_settings_and_fresh_dir(tmp_path, capsys):
    out = tmp_path / "logs" / "run-001"  # parents are created as needed
    factory, created = recording_factory()
    fake = make_fake_ppo()

    assert tq.main(argv_for(out, timesteps="2048"), ppo_cls=fake, env_factory=factory) == 0

    (model,) = fake.created
    assert fake.loads == []
    assert model.policy_name == "MultiInputPolicy"
    assert model.kwargs == {
        **tq.FIXED_HYPERPARAMETERS,
        "n_steps": 512,
        "n_epochs": 20,
        "seed": 0,
        "device": "auto",
        "tensorboard_log": None,
        "verbose": 1,
    }
    assert isinstance(model.env, DummyVecEnv)
    assert model.env.envs[0].env._max_episode_steps == 1024

    (call,) = model.learn_calls
    assert call["total_timesteps"] == 2048
    assert call["progress_bar"] is False
    assert call["reset_num_timesteps"] is True
    callback = call["callback"]
    assert isinstance(callback, EvalCallback)
    assert callback.eval_freq == 10_240 and callback.n_eval_episodes == 5
    assert callback.deterministic is True and callback.render is False
    assert Path(callback.best_model_save_path) == out / "best_model"

    assert model.saved == [out / tq.CHECKPOINT_NAME]
    assert (out / tq.CHECKPOINT_NAME).is_file()
    assert (out / tq.RUN_CONFIG_NAME).is_file()
    assert (out / "train.monitor.csv").is_file()
    assert (out / "eval.monitor.csv").is_file()
    assert len(created) == 2 and all(env.closed for env in created)
    assert str(out / tq.CHECKPOINT_NAME) in capsys.readouterr().out


def test_smoke_flags_are_explicit_opt_in(tmp_path):
    fake = make_fake_ppo()
    argv = argv_for(
        tmp_path / "smoke",
        "--n-steps", "64",
        "--n-epochs", "1",
        "--eval-freq", "64",
        "--eval-episodes", "1",
        "--seed", "7",
        "--device", "cpu",
    )
    assert tq.main(argv, ppo_cls=fake, env_factory=recording_factory()[0]) == 0
    (model,) = fake.created
    assert model.kwargs["n_steps"] == 64 and model.kwargs["n_epochs"] == 1
    assert model.kwargs["seed"] == 7 and model.kwargs["device"] == "cpu"
    assert model.kwargs["learning_rate"] == 1e-4  # other settings unchanged
    callback = model.learn_calls[0]["callback"]
    assert callback.eval_freq == 64 and callback.n_eval_episodes == 1


def test_global_rng_is_restored(tmp_path, seeded_global_rng):
    before = rng_snapshot()
    fake = make_fake_ppo()  # its learn() perturbs random/numpy/torch
    assert tq.main(argv_for(tmp_path / "run"), ppo_cls=fake,
                   env_factory=recording_factory()[0]) == 0
    assert_same_rng(before, rng_snapshot())


def test_envs_closed_and_rng_restored_when_training_fails(tmp_path, seeded_global_rng):
    out = tmp_path / "run"
    factory, created = recording_factory()
    fake = make_fake_ppo(learn_error=RuntimeError("boom"))
    before = rng_snapshot()

    with pytest.raises(RuntimeError, match="boom"):
        tq.main(argv_for(out), ppo_cls=fake, env_factory=factory)

    assert len(created) == 2 and all(env.closed for env in created)
    assert not (out / tq.CHECKPOINT_NAME).exists()
    assert_same_rng(before, rng_snapshot())


# --------------------------------------------------------------------------- #
# Output directory safety
# --------------------------------------------------------------------------- #
def test_existing_output_dir_is_rejected_untouched(tmp_path, capsys):
    out = tmp_path / "run"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("previous run")
    factory, created = recording_factory()

    assert tq.main(argv_for(out), ppo_cls=ExplodingPPO, env_factory=factory) == 1

    assert "already exists" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
    assert sentinel.read_text() == "previous run"
    assert created == []


@pytest.mark.parametrize(
    "target", [tq.COMMITTED_CHECKPOINT, REPO_ROOT / "rl", REPO_ROOT / "rl" / "logs"]
)
def test_repo_paths_are_never_reused(target, capsys):
    if not target.exists():
        pytest.skip(f"{target} does not exist in this checkout")
    before = rl_dir_listing()
    assert tq.main(argv_for(target), ppo_cls=ExplodingPPO,
                   env_factory=recording_factory()[0]) == 1
    assert "already exists" in capsys.readouterr().err
    assert rl_dir_listing() == before


def test_output_dir_under_a_file_is_rejected(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert tq.main(argv_for(blocker / "run"), ppo_cls=ExplodingPPO,
                   env_factory=recording_factory()[0]) == 1
    assert "not a directory" in capsys.readouterr().err
    assert blocker.read_text() == "x"


# --------------------------------------------------------------------------- #
# Resume: explicit, trusted, compatible, fail-closed
# --------------------------------------------------------------------------- #
def test_missing_resume_fails_closed(tmp_path, capsys):
    out = tmp_path / "run"
    factory, created = recording_factory()
    argv = argv_for(out, "--resume", str(tmp_path / "missing.zip"))

    assert tq.main(argv, ppo_cls=ExplodingPPO, env_factory=factory) == 1

    assert "resume checkpoint not found" in capsys.readouterr().err
    assert not out.exists()
    assert created == []


def test_corrupt_resume_fails_closed(tmp_path, capsys):
    out = tmp_path / "run"
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a checkpoint")
    factory, created = recording_factory()

    assert tq.main(argv_for(out, "--resume", str(bad)), env_factory=factory) == 1

    assert "could not load PPO checkpoint" in capsys.readouterr().err
    assert not out.exists()
    assert created == []


def test_incompatible_resume_fails_closed(tmp_path, capsys, env_spaces):
    out = tmp_path / "run"
    checkpoint = tmp_path / "trusted.zip"
    checkpoint.write_bytes(b"placeholder; the fake loader ignores it")
    loaded = LoadedFakeModel(env_spaces[0], spaces.Discrete(4))
    fake = make_fake_ppo(loaded=loaded)
    factory, created = recording_factory()

    assert tq.main(argv_for(out, "--resume", str(checkpoint)),
                   ppo_cls=fake, env_factory=factory) == 1

    assert "action space does not match" in capsys.readouterr().err
    assert not out.exists()
    assert fake.created == [] and loaded.learn_calls == []
    assert len(created) == 1 and created[0].closed  # reference env closed


def test_real_incompatible_checkpoint_fails_closed(tmp_path, capsys):
    import gymnasium as gym
    from stable_baselines3 import PPO

    cartpole = gym.make("CartPole-v1")
    try:
        PPO("MlpPolicy", cartpole, n_steps=64, batch_size=64, device="cpu").save(
            str(tmp_path / "cartpole.zip")
        )
    finally:
        cartpole.close()

    out = tmp_path / "run"
    argv = argv_for(out, "--resume", str(tmp_path / "cartpole.zip"), "--device", "cpu")
    assert tq.main(argv) == 1
    assert "observation space does not match" in capsys.readouterr().err
    assert not out.exists()


def test_resume_loads_full_checkpoint_and_ignores_stored_log_path(tmp_path, env_spaces):
    out = tmp_path / "resume"
    checkpoint = tmp_path / "trusted.zip"
    checkpoint.write_bytes(b"placeholder; the fake loader ignores it")
    loaded = LoadedFakeModel(*env_spaces)
    fake = make_fake_ppo(loaded=loaded)
    factory, created = recording_factory()
    argv = argv_for(out, "--resume", str(checkpoint), "--seed", "5", "--device", "cpu")

    assert tq.main(argv, ppo_cls=fake, env_factory=factory) == 0

    assert fake.loads == [(str(checkpoint.absolute()), "cpu")]
    assert fake.created == []  # no fresh model, no weight cherry-picking
    (vec_env,) = loaded.set_env_calls
    assert isinstance(vec_env, DummyVecEnv)
    assert loaded.seeds == [5]
    assert loaded.tensorboard_log is None
    (call,) = loaded.learn_calls
    assert call["reset_num_timesteps"] is False and call["progress_bar"] is False
    assert loaded.saved == [out / tq.CHECKPOINT_NAME]
    assert all(env.closed for env in created)


def test_committed_checkpoint_is_resume_compatible_read_only():
    from stable_baselines3 import PPO

    model = tq.load_checkpoint(tq.COMMITTED_CHECKPOINT, device="cpu", ppo_cls=PPO)
    env = FullQuestEnv()
    try:
        tq.check_compatibility(model, env)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# One tiny real run (pipeline check only, not a quality claim)
# --------------------------------------------------------------------------- #
def test_tiny_real_training_run_is_seeded_and_contained(tmp_path, seeded_global_rng):
    from stable_baselines3 import PPO

    smoke = [
        "--timesteps", "64", "--n-steps", "64", "--n-epochs", "1",
        "--eval-freq", "64", "--eval-episodes", "1", "--seed", "3", "--device", "cpu",
    ]
    before = rng_snapshot()
    runs = [tmp_path / "logs" / "a", tmp_path / "logs" / "b"]
    for out in runs:
        assert tq.main(["--output-dir", str(out), *smoke]) == 0
    assert_same_rng(before, rng_snapshot())

    for out in runs:
        assert (out / tq.CHECKPOINT_NAME).is_file()
        assert (out / "best_model" / "best_model.zip").is_file()
        assert (out / "evaluations.npz").is_file()
        assert not (out / "tensorboard").exists()

    models = [PPO.load(str(out / tq.CHECKPOINT_NAME), device="cpu") for out in runs]
    env = FullQuestEnv()
    try:
        tq.check_compatibility(models[0], env)
    finally:
        env.close()
    assert models[0].n_steps == 64 and models[0].n_epochs == 1
    state_a, state_b = (m.policy.state_dict() for m in models)
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key
