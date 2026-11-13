"""Tests for seeded FullQuestEnv resets and the headless PPO evaluator.

These tests never open a window, download a model, call an LLM or train.
Orchestration tests use a tiny deterministic fake model; the real committed
checkpoint is evaluated separately via ``python rl/run_full_quest.py``.
"""

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402
from gymnasium import spaces  # noqa: E402

from configs import config  # noqa: E402
from game.games.game import Game  # noqa: E402
from rl import run_full_quest as rq  # noqa: E402
from rl.environments.full_quest_env_pass import FullQuestEnv  # noqa: E402

OBS_KEYS = {
    "grid",
    "has_location_info",
    "visited_treasure_first",
    "has_password_info",
    "password_input_mode",
    "treasure_opened",
    "treasure_password",
    "entered_password",
}


def layout(game):
    return (
        tuple(tuple(row) for row in game.grid),
        tuple(game.player_pos),
        tuple(game.exit_pos),
        tuple(game.treasure_pos),
        tuple(tuple(npc.pos) for npc in game.npcs),
        game.password,
    )


@pytest.fixture
def env():
    e = FullQuestEnv()
    try:
        yield e
    finally:
        e.close()


class FakeModel:
    """Deterministic stand-in for PPO: always returns the same action."""

    def __init__(self, action=4, observation_space=None, action_space=None):
        reference = FullQuestEnv()
        self.observation_space = (
            reference.observation_space if observation_space is None else observation_space
        )
        self.action_space = reference.action_space if action_space is None else action_space
        reference.close()
        self.action = action
        self.deterministic_flags = []

    def predict(self, obs, deterministic=False):
        self.deterministic_flags.append(deterministic)
        return np.array(self.action), None


class TerminatingEnv(FullQuestEnv):
    """Test double that reports ``terminated`` after N steps.

    Used only to check that the evaluator distinguishes termination from
    truncation; it says nothing about whether the real policy solves quests.
    """

    terminate_at = 3

    def reset(self, seed=None, options=None):
        self._count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, _terminated, truncated, info = super().step(action)
        self._count += 1
        return obs, reward, self._count >= self.terminate_at, truncated, info


# --------------------------------------------------------------------------- #
# Seeded reset reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_reproduces_layout_and_observation(env):
    obs_a, _ = env.reset(seed=77)
    layout_a = layout(env.game)

    obs_b, _ = env.reset(seed=77)
    assert layout(env.game) == layout_a
    assert np.array_equal(obs_a["grid"], obs_b["grid"])

    other = FullQuestEnv()
    try:
        other.reset(seed=1)
        for action in (0, 3, 4, 1):
            other.step(action)
        obs_c, _ = other.reset(seed=77)
        assert layout(other.game) == layout_a
        assert np.array_equal(obs_a["grid"], obs_c["grid"])
    finally:
        other.close()


def test_different_seeds_give_different_layouts(env):
    layouts = set()
    for seed in range(10):
        env.reset(seed=seed)
        layouts.add(layout(env.game))
    assert len(layouts) > 1


def test_seeded_reset_does_not_touch_global_random():
    saved = random.getstate()
    try:
        random.seed(2024)
        before = random.getstate()
        e = FullQuestEnv()
        e.reset(seed=5)
        e.reset()  # continues the seeded stream, not the global one
        e.close()
        assert random.getstate() == before
    finally:
        random.setstate(saved)


def test_unseeded_resets_after_seed_follow_seeded_stream():
    def sequence():
        e = FullQuestEnv()
        try:
            e.reset(seed=3)
            first = layout(e.game)
            e.reset()
            second = layout(e.game)
            return first, second
        finally:
            e.close()

    assert sequence() == sequence()


def test_game_without_rng_still_uses_global_random():
    saved = random.getstate()
    try:
        random.seed(123)
        a = layout(Game(llm_client=None))
        random.seed(123)
        b = layout(Game(llm_client=None))
        assert a == b

        random.seed(123)
        first_draw = random.random()
        random.seed(123)
        g = Game(llm_client=None)
        assert random.random() != first_draw  # global stream was consumed
        g.reset()
        assert layout(g)  # plain reset() still works for existing callers
    finally:
        random.setstate(saved)


def test_env_works_from_other_cwd(tmp_path, monkeypatch):
    assert os.path.isabs(config.NPC_LOC_PROMPT_PATH)
    assert os.path.isabs(config.NPC_PW_PROMPT_PATH)
    monkeypatch.chdir(tmp_path)
    e = FullQuestEnv()
    try:
        obs, _ = e.reset(seed=9)
        assert obs["grid"].shape == (config.GRID_HEIGHT, config.GRID_WIDTH, 6)
    finally:
        e.close()


# --------------------------------------------------------------------------- #
# Observation / action space contract (must stay checkpoint-compatible)
# --------------------------------------------------------------------------- #
def test_observation_and_action_space_contract(env):
    assert env.action_space == spaces.Discrete(15)
    assert isinstance(env.observation_space, spaces.Dict)
    assert set(env.observation_space.spaces) == OBS_KEYS

    grid_space = env.observation_space["grid"]
    assert grid_space.shape == (config.GRID_HEIGHT, config.GRID_WIDTH, 6)
    assert grid_space.dtype == np.uint8

    for key, n in {
        "has_location_info": 2,
        "visited_treasure_first": 2,
        "has_password_info": 2,
        "password_input_mode": 2,
        "treasure_opened": 2,
        "treasure_password": 10,
        "entered_password": 10,
    }.items():
        assert env.observation_space[key] == spaces.Discrete(n)

    obs, info = env.reset(seed=11)
    assert info == {}
    assert env.observation_space.contains(obs)
    for action in range(15):
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(terminated, bool) and truncated is False


# --------------------------------------------------------------------------- #
# Evaluator orchestration
# --------------------------------------------------------------------------- #
def test_evaluate_truncates_at_max_steps_with_fake_model():
    model = FakeModel(action=4)
    result = rq.evaluate(model, episodes=2, base_seed=10, max_steps=5)

    assert result["max_steps"] == 5
    assert [r["seed"] for r in result["results"]] == [10, 11]
    for r in result["results"]:
        assert r["outcome"] == "truncated"
        assert r["truncated"] is True and r["terminated"] is False
        assert r["steps"] == 5
        assert r["milestone_first_step"]["exit_reached"] is None
        assert set(r["milestone_first_step"]) == set(rq.MILESTONES)
    assert result["summary"]["solved"] == 0
    assert result["summary"]["truncated"] == 2
    assert model.deterministic_flags and all(model.deterministic_flags)
    json.dumps(result)  # machine-readable


def test_evaluate_reports_termination_distinct_from_truncation():
    factory = lambda render_mode=None: TerminatingEnv(render_mode=render_mode)  # noqa: E731

    result = rq.evaluate(
        FakeModel(action=4), episodes=1, base_seed=0, max_steps=10, env_factory=factory
    )
    (r,) = result["results"]
    assert r["outcome"] == "solved"
    assert r["terminated"] is True and r["truncated"] is False
    assert r["steps"] == TerminatingEnv.terminate_at
    # Milestones come from env state, not from the terminated flag.
    assert r["milestone_first_step"]["treasure_opened"] is None

    # Terminated on the same step the time limit hits: termination wins.
    result = rq.evaluate(
        FakeModel(action=4),
        episodes=1,
        base_seed=0,
        max_steps=TerminatingEnv.terminate_at,
        env_factory=factory,
    )
    (r,) = result["results"]
    assert r["terminated"] is True and r["truncated"] is True
    assert r["outcome"] == "solved"


def test_classify():
    assert rq.classify(True, False) == "solved"
    assert rq.classify(True, True) == "solved"
    assert rq.classify(False, True) == "truncated"


def test_seeded_evaluation_is_reproducible():
    a = rq.evaluate(FakeModel(action=0), episodes=2, base_seed=4, max_steps=6)
    b = rq.evaluate(FakeModel(action=0), episodes=2, base_seed=4, max_steps=6)
    assert a == b


def test_env_closed_even_when_policy_raises():
    created = []

    class RecordingEnv(FullQuestEnv):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def factory(render_mode=None):
        e = RecordingEnv(render_mode=render_mode)
        created.append(e)
        return e

    class BrokenModel(FakeModel):
        def predict(self, obs, deterministic=False):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        rq.evaluate(BrokenModel(), episodes=1, base_seed=0, max_steps=3, env_factory=factory)
    assert created and created[0].closed


def test_incompatible_model_fails_visibly():
    with pytest.raises(rq.EvaluationError, match="action space"):
        rq.evaluate(
            FakeModel(action_space=spaces.Discrete(4)), episodes=1, base_seed=0, max_steps=3
        )
    with pytest.raises(rq.EvaluationError, match="observation space"):
        rq.evaluate(
            FakeModel(observation_space=spaces.Discrete(3)),
            episodes=1,
            base_seed=0,
            max_steps=3,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": 0},
        {"max_steps": 0},
        {"base_seed": -1},
        {"delay": float("nan")},
        {"delay": 0.5},  # delay without render
    ],
)
def test_evaluate_rejects_invalid_arguments(kwargs):
    params = {"episodes": 1, "base_seed": 0, "max_steps": 3, **kwargs}
    with pytest.raises(ValueError):
        rq.evaluate(FakeModel(), **params)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_defaults_match_training_and_are_repo_anchored():
    args = rq.parse_args([])
    assert args.max_steps == rq.TRAINING_MAX_STEPS == 1024
    assert args.render is False and args.delay == 0.0
    assert args.model == rq.DEFAULT_MODEL_PATH
    assert rq.DEFAULT_MODEL_PATH.is_absolute()
    assert rq.DEFAULT_MODEL_PATH == REPO_ROOT / "rl" / "ppo_full_quest.zip"


@pytest.mark.parametrize(
    "argv",
    [
        ["--episodes", "0"],
        ["--episodes", "-2"],
        ["--episodes", "two"],
        ["--max-steps", "0"],
        ["--max-steps", "1.5"],
        ["--seed", "-1"],
        ["--delay", "nan"],
        ["--delay", "-1"],
        ["--delay", "0.5"],  # requires --render
    ],
)
def test_cli_rejects_invalid_input(argv):
    with pytest.raises(SystemExit) as excinfo:
        rq.main(argv)
    assert excinfo.value.code == 2


def test_cli_missing_checkpoint_fails(tmp_path, capsys):
    code = rq.main(["--episodes", "1", "--model", str(tmp_path / "missing.zip")])
    assert code == 1
    assert "checkpoint not found" in capsys.readouterr().err


def test_cli_corrupt_checkpoint_fails(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a checkpoint")
    code = rq.main(["--episodes", "1", "--model", str(bad)])
    assert code == 1
    assert "could not load PPO checkpoint" in capsys.readouterr().err


def test_cli_headless_report(monkeypatch, capsys, tmp_path):
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return FakeModel(action=1)

    def no_sleep(_seconds):
        raise AssertionError("headless mode must not sleep")

    monkeypatch.setattr(rq, "load_model", fake_load)
    monkeypatch.setattr(rq.time, "sleep", no_sleep)
    monkeypatch.chdir(tmp_path)

    code = rq.main(["--episodes", "2", "--seed", "5", "--max-steps", "4"])
    assert code == 0
    assert loaded == [rq.DEFAULT_MODEL_PATH]

    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"] == "rl/ppo_full_quest.zip"
    assert report["deterministic"] is True
    assert report["max_steps"] == 4
    assert [r["seed"] for r in report["results"]] == [5, 6]
    assert all(r["outcome"] == "truncated" for r in report["results"])
    assert report["summary"]["solved"] + report["summary"]["truncated"] == 2
    assert not pygame.display.get_init()
