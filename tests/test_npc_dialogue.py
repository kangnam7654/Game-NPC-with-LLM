"""Offline tests for the NPC dialogue contract.

Nothing here needs Ollama, a model download or network access. The Ollama
client is exercised with deterministic fakes of ``requests.post`` and a small
HTTP server on 127.0.0.1. The game tests drive the real ``InputHandler`` /
``InteractionHandler`` with posted Pygame events under the SDL dummy video
driver, using a scripted fake LLM client whose replies are synthetic.
"""

import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pygame  # noqa: E402
import pytest  # noqa: E402
import requests  # noqa: E402

from configs import config  # noqa: E402
from game.controllers import interaction_handler as ih  # noqa: E402
from game.controllers.input_handler import InputHandler  # noqa: E402
from game.games.game import Game  # noqa: E402
from game.games.states import GameState  # noqa: E402
from game.renderers.renderer import Renderer  # noqa: E402
from llm.clients import ollama_client  # noqa: E402
from llm.clients.ollama_client import OllamaClient  # noqa: E402

SECRET_USER = "USER-SECRET-7f3a"
SECRET_SYSTEM = "SYSTEM-SECRET-91bc"


# =========================================================================== #
# OllamaClient with a fake requests.post
# =========================================================================== #
def make_response(status, body: bytes, content_type="application/json"):
    response = requests.models.Response()
    response.status_code = status
    response._content = body
    response.headers["Content-Type"] = content_type
    response.encoding = "utf-8"
    response.url = "http://localhost:11434/api/chat"
    return response


def install_post(monkeypatch, result):
    calls = []

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append({"url": url, "json": json, "timeout": timeout, **kwargs})
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    return calls


def secret_messages():
    return [{"role": "user", "content": SECRET_USER}]


def test_defaults_preserve_local_endpoint_and_model():
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.model is None
    assert client.timeout == ollama_client.DEFAULT_TIMEOUT
    assert client.timeout is not None
    with pytest.raises(ValueError):
        OllamaClient(timeout=None)


def test_success_returns_message_content_with_bounded_timeout(monkeypatch):
    body = json.dumps(
        {
            "model": "gpt-oss:20b",
            "message": {"role": "assistant", "content": "안녕하신가"},
            "done": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    calls = install_post(monkeypatch, make_response(200, body))
    client = OllamaClient(model="gpt-oss:20b")

    reply = client.chat(
        [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q"}],
        system_prompt="sys",
    )

    assert reply == "안녕하신가"
    assert client.last_error is None
    (call,) = calls
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["timeout"] == ollama_client.DEFAULT_TIMEOUT
    payload = call["json"]
    assert payload["model"] == "gpt-oss:20b"
    assert payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1:] == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "q"},
    ]


def test_leading_system_message_is_replaced(monkeypatch):
    body = b'{"message": {"role": "assistant", "content": "ok"}}'
    calls = install_post(monkeypatch, make_response(200, body))
    client = OllamaClient(model="m")
    client.chat([{"role": "system", "content": "old"}, {"role": "user", "content": "q"}])
    messages = calls[0]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": client.default_system_prompt()}
    assert [m["content"] for m in messages].count("old") == 0


@pytest.mark.parametrize(
    "body",
    [
        b'{"message": null}',
        b"{}",
        b"[]",
        b"null",
        b'{"message": "text"}',
        b'{"message": {"role": "assistant"}}',
        b'{"message": {"content": null}}',
        b'{"message": {"content": 5}}',
        b'{"message": {"content": ""}}',
        b'{"message": {"content": "   \\n "}}',
        b"not json at all",
        b"",
        # NDJSON (what a streaming reply looks like) is not one JSON document.
        b'{"message": {"content": "a"}, "done": false}\n'
        b'{"message": {"content": "b"}, "done": true}\n',
    ],
)
def test_missing_or_malformed_body_returns_failure_sentinel(monkeypatch, capsys, body):
    install_post(monkeypatch, make_response(200, body))
    client = OllamaClient(model="m")
    assert client.chat(secret_messages(), system_prompt=SECRET_SYSTEM) is None
    assert client.last_error
    out = capsys.readouterr().out
    assert SECRET_USER not in out and SECRET_SYSTEM not in out


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_http_failure_returns_none_without_echoing_secrets(monkeypatch, capsys, status):
    echoed = f'{{"error": "bad request: {SECRET_USER} {SECRET_SYSTEM}"}}'.encode()
    install_post(monkeypatch, make_response(status, echoed))
    client = OllamaClient(model="m")

    assert client.chat(secret_messages(), system_prompt=SECRET_SYSTEM) is None
    assert str(status) in client.last_error
    captured = capsys.readouterr()
    for output in (captured.out, captured.err, client.last_error):
        assert SECRET_USER not in output and SECRET_SYSTEM not in output


@pytest.mark.parametrize(
    "error, reason",
    [
        (requests.exceptions.ReadTimeout("read timed out"), "timed out"),
        (requests.exceptions.ConnectTimeout("connect timed out"), "timed out"),
        (requests.exceptions.ConnectionError(f"refused {SECRET_USER}"), "connect"),
        (requests.exceptions.TooManyRedirects(f"loop {SECRET_USER}"), "TooManyRedirects"),
    ],
)
def test_transport_errors_return_none(monkeypatch, capsys, error, reason):
    install_post(monkeypatch, error)
    client = OllamaClient(model="m")
    assert client.chat(secret_messages(), system_prompt=SECRET_SYSTEM) is None
    assert reason in client.last_error
    out = capsys.readouterr().out
    assert SECRET_USER not in out and SECRET_USER not in client.last_error


def test_streaming_is_rejected_not_misparsed(monkeypatch):
    calls = install_post(monkeypatch, make_response(200, b"{}"))
    with pytest.raises(ValueError, match="stream"):
        OllamaClient(model="m").chat(secret_messages(), stream=True)
    assert calls == []


# =========================================================================== #
# OllamaClient against a real loopback HTTP server
# =========================================================================== #
class _OllamaLikeHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.paths.append(self.path)
        if self.server.slow:
            self.server.release.wait(5)
        status, body = self.server.reply
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # client already gave up (timeout test)

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback(monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaLikeHandler)
    server.daemon_threads = True
    server.paths = []
    server.slow = False
    server.release = threading.Event()
    server.reply = (200, b'{"message": {"role": "assistant", "content": "loop ok"}}')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()


def test_loopback_success_and_http_error(loopback):
    port = loopback.server_address[1]
    client = OllamaClient(host="127.0.0.1", port=port, model="m", timeout=(2.0, 2.0))
    assert client.chat([{"role": "user", "content": "q"}]) == "loop ok"
    assert loopback.paths == ["/api/chat"]

    loopback.reply = (500, b'{"error": "boom"}')
    assert client.chat([{"role": "user", "content": "q"}]) is None
    assert "500" in client.last_error


def test_loopback_read_timeout_is_enforced(loopback):
    loopback.slow = True
    port = loopback.server_address[1]
    client = OllamaClient(host="127.0.0.1", port=port, model="m", timeout=(2.0, 0.3))

    started = time.monotonic()
    assert client.chat([{"role": "user", "content": "q"}]) is None
    elapsed = time.monotonic() - started

    assert client.last_error == "request timed out"
    assert elapsed < 4.0


# =========================================================================== #
# Headless Pygame interaction
# =========================================================================== #
class FakeLLM:
    """Scripted stand-in for OllamaClient. Replies are popped in order."""

    def __init__(self):
        self.replies = []
        self.calls = []

    def chat(self, messages, stream=False, model=None, system_prompt=None):
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "system_prompt": system_prompt,
                "stream": stream,
            }
        )
        reply = self.replies.pop(0) if self.replies else None
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def ctx():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    llm = FakeLLM()
    game = Game(llm_client=llm, rng=random.Random(1234))
    renderer = Renderer(screen)
    interaction = ih.InteractionHandler(game)
    inputs = InputHandler(game, interaction, renderer)
    loc_npc, pw_npc = game.npcs
    assert loc_npc.name == ih.LOCATION_NPC_NAME and pw_npc.name == ih.PASSWORD_NPC_NAME
    x, y = game.treasure_pos
    try:
        yield SimpleNamespace(
            game=game,
            llm=llm,
            renderer=renderer,
            inputs=inputs,
            loc=loc_npc,
            pw=pw_npc,
            x=x,
            y=y,
            wrong_y=(y + 1) % config.GRID_HEIGHT,
            password=game.password,
            wrong_password=str((int(game.password) + 1) % 10),
        )
    finally:
        pygame.key.stop_text_input()
        pygame.quit()


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)


def text(s):
    return pygame.event.Event(pygame.TEXTINPUT, text=s)


def send(ctx, *events):
    pygame.event.clear()
    for event in events:
        pygame.event.post(event)
    assert ctx.inputs.handle_events() is True


def open_menu(ctx, npc):
    """Walk up to ``npc`` (other NPCs moved out of reach) and press Space."""
    game = ctx.game
    send(ctx, key(pygame.K_ESCAPE))  # leave any open chat/menu
    game.state = GameState.PLAYING
    game.player_pos = npc.pos
    corners = [
        (0, 0),
        (config.GRID_WIDTH - 1, 0),
        (0, config.GRID_HEIGHT - 1),
        (config.GRID_WIDTH - 1, config.GRID_HEIGHT - 1),
    ]
    far = max(corners, key=lambda c: abs(c[0] - npc.pos[0]) + abs(c[1] - npc.pos[1]))
    for other in game.npcs:
        if other is not npc and game.is_adjacent(npc.pos, other.pos):
            other.pos = far
    send(ctx, key(pygame.K_SPACE))
    assert game.state == GameState.INTERACTION_MENU
    assert game.active_npc is npc
    assert game.menu_selection == 0


def open_chat(ctx, npc):
    open_menu(ctx, npc)
    send(ctx, key(pygame.K_DOWN), key(pygame.K_RETURN))
    assert ctx.game.state == GameState.TEXT_INPUT
    assert npc.chat_history == [{"role": "assistant", "content": "무엇이 궁금한가?"}]


def say(ctx, message, reply):
    ctx.llm.replies.append(reply)
    send(ctx, text(message), key(pygame.K_RETURN))


def use_fixed_menu(ctx, npc):
    open_menu(ctx, npc)
    send(ctx, key(pygame.K_RETURN))
    assert ctx.game.state == GameState.PLAYING


# --- Location -------------------------------------------------------------
@pytest.mark.parametrize(
    "template",
    [
        "보물은 ({x}, {y}) 좌표에 숨겨져 있다네.",
        "자네가 찾는 보물의 좌표는 {x}, {y}이네.",
        "보물은 x={x}, y={y} 지점에 있지!",
        "보물이 어디 있냐고? [{x},{y}]에 있다네.",
    ],
)
def test_explicit_correct_coordinate_pair_unlocks_location(ctx, template):
    open_chat(ctx, ctx.loc)
    reply = template.format(x=ctx.x, y=ctx.y)
    say(ctx, "보물은 어디에 있나요?", reply)

    game = ctx.game
    assert game.knows_location is True
    assert game.knows_password is False
    assert ctx.loc.chat_history[-2:] == [
        {"role": "user", "content": "보물은 어디에 있나요?"},
        {"role": "assistant", "content": reply},
    ]
    assert game.input_text == ""
    assert game.state == GameState.TEXT_INPUT


@pytest.mark.parametrize(
    "template",
    [
        # wrong y
        "보물은 ({x}, {wy}) 좌표에 있네.",
        # incidental digits (the base code granted location on the x digit alone)
        "보물을 찾던 {x}명의 모험가가 {y}일 동안 헤맸다네.",
        "보물은 x={x} 줄 어딘가에 있지.",
        # a pair, but no treasure/location context
        "오늘은 ({x}, {y})번 탁자에서 차를 마셨네.",
        # negated / question / ambiguous
        "보물은 ({x}, {y})에는 없다네.",
        "보물이 ({x}, {y})에 있다고 생각하나?",
        "보물은 ({x}, {y}) 또는 ({x}, {wy}) 중 한 곳에 있네.",
        # meta-discussion with no answer
        "좌표를 묻는구먼. 그건 아직 알려줄 수 없네.",
    ],
)
def test_wrong_or_incidental_numbers_do_not_unlock_location(ctx, template):
    open_chat(ctx, ctx.loc)
    say(ctx, "보물은 어디에 있나요?", template.format(x=ctx.x, y=ctx.y, wy=ctx.wrong_y))
    assert ctx.game.knows_location is False
    assert ctx.game.knows_password is False


def test_echo_of_players_own_guess_does_not_unlock_location(ctx):
    open_chat(ctx, ctx.loc)
    say(
        ctx,
        f"보물이 ({ctx.x}, {ctx.y})에 있나요",
        f"그렇다네, 보물은 ({ctx.x}, {ctx.y})에 있네.",
    )
    assert ctx.game.knows_location is False


def test_wrong_asserted_pair_still_blocks_when_player_supplied_it(ctx):
    open_chat(ctx, ctx.loc)
    say(
        ctx,
        f"보물이 ({ctx.x}, {ctx.wrong_y})인가요",
        f"보물은 ({ctx.x}, {ctx.y}) 또는 ({ctx.x}, {ctx.wrong_y}) 중 하나라네.",
    )
    assert ctx.game.knows_location is False


def test_password_npc_cannot_grant_location(ctx):
    open_chat(ctx, ctx.pw)
    say(ctx, "보물은 어디에?", f"보물은 ({ctx.x}, {ctx.y})에 있다네.")
    assert ctx.game.knows_location is False


# --- Password -------------------------------------------------------------
def test_password_is_gated_until_location_is_known(ctx):
    game = ctx.game
    open_chat(ctx, ctx.pw)
    say(ctx, "암호를 알려주세요.", f"암호는 '{ctx.password}'일세.")

    assert game.knows_password is False
    locked_prompt = ctx.llm.calls[-1]["system_prompt"]
    assert ih.LOCKED_PASSWORD_PLACEHOLDER in locked_prompt
    assert not re.search(r"\d", locked_prompt)  # the secret is not sent at all

    use_fixed_menu(ctx, ctx.loc)  # learn the location the reliable way
    assert game.knows_location is True

    open_chat(ctx, ctx.pw)
    say(ctx, "위치를 알아왔습니다. 암호를 알려주세요.", f"좋네, 암호는 '{ctx.password}'일세.")
    assert game.knows_password is True
    assert ctx.llm.calls[-1]["system_prompt"] == ctx.pw.background
    assert ctx.password in ctx.pw.background


@pytest.mark.parametrize(
    "template",
    [
        "암호는 '{wrong}'일세.",
        "암호에 대해서는 말할 수 없네. 다만 {pw}개의 별을 보게.",
        "{pw}번째 계단을 오르면 지혜를 얻으리.",
        "암호는 '{pw}'이 아닐세.",
        "암호가 {pw}라고 생각하는가?",
        "암호는 {pw} 아니면 {wrong}일세.",
    ],
)
def test_wrong_or_incidental_password_text_does_not_grant(ctx, template):
    use_fixed_menu(ctx, ctx.loc)
    open_chat(ctx, ctx.pw)
    say(
        ctx,
        "암호를 알려주세요.",
        template.format(pw=ctx.password, wrong=ctx.wrong_password),
    )
    assert ctx.game.knows_location is True
    assert ctx.game.knows_password is False


def test_password_echo_of_player_guess_does_not_grant(ctx):
    use_fixed_menu(ctx, ctx.loc)
    open_chat(ctx, ctx.pw)
    say(ctx, f"암호가 {ctx.password}인가요", f"그렇다네, 암호는 {ctx.password}일세.")
    assert ctx.game.knows_password is False


def test_location_npc_cannot_grant_password(ctx):
    open_chat(ctx, ctx.loc)
    say(
        ctx,
        "보물과 암호를 알려주세요.",
        f"보물은 ({ctx.x}, {ctx.y})에 있네. 암호는 '{ctx.password}'일세.",
    )
    assert ctx.game.knows_location is True
    assert ctx.game.knows_password is False


# --- Failed chat ----------------------------------------------------------
@pytest.mark.parametrize(
    "bad_reply",
    [
        None,
        "",
        "   ",
        "\U0001f600",
        "<think>hmm</think>",
        {"message": None},
        42,
        RuntimeError("transport exploded"),
    ],
)
def test_failed_chat_keeps_state_and_allows_retry(ctx, bad_reply):
    game = ctx.game
    open_chat(ctx, ctx.loc)
    question = "보물은 어디에 있나요?"
    say(ctx, question, bad_reply)

    assert ctx.loc.chat_history == [{"role": "assistant", "content": "무엇이 궁금한가?"}]
    assert game.input_text == question  # kept for retry
    assert game.dialogue == ih.CHAT_FAILURE_NOTICE
    assert game.knows_location is False and game.knows_password is False
    assert game.state == GameState.TEXT_INPUT and game.active_npc is ctx.loc
    ctx.renderer.draw(game)  # the notice renders headlessly

    good = f"보물은 ({ctx.x}, {ctx.y})에 있다네."
    ctx.llm.replies.append(good)
    send(ctx, key(pygame.K_RETURN))  # retry the same text

    assert ctx.loc.chat_history[-2:] == [
        {"role": "user", "content": question},
        {"role": "assistant", "content": good},
    ]
    assert game.input_text == ""
    assert game.dialogue != ih.CHAT_FAILURE_NOTICE
    assert game.knows_location is True
    retried = ctx.llm.calls[-1]["messages"]
    assert [m["role"] for m in retried] == ["assistant", "user"]
    assert all(m["content"] != "..." for m in retried)


def test_missing_client_is_a_retryable_failure(ctx):
    ctx.game.llm_client = None
    open_chat(ctx, ctx.loc)
    say(ctx, "안녕하세요", "unused")
    assert ctx.game.dialogue == ih.CHAT_FAILURE_NOTICE
    assert ctx.game.input_text == "안녕하세요"
    assert len(ctx.loc.chat_history) == 1


def test_empty_input_does_not_call_llm(ctx):
    open_chat(ctx, ctx.loc)
    send(ctx, text("   "), key(pygame.K_RETURN))
    assert ctx.llm.calls == []
    assert len(ctx.loc.chat_history) == 1


def test_chat_history_stays_bounded(ctx):
    open_chat(ctx, ctx.loc)
    for i in range(10):
        say(ctx, f"질문 {i}", f"대답 {i}")
        assert len(ctx.loc.chat_history) <= ih.MAX_CHAT_HISTORY
    assert ctx.loc.chat_history[-1] == {"role": "assistant", "content": "대답 9"}
    assert max(len(c["messages"]) for c in ctx.llm.calls) <= ih.MAX_CHAT_HISTORY + 1
    assert ctx.game.knows_location is False


# --- Fixed-information menu (no LLM) ---------------------------------------
def test_fixed_menu_behaviour_is_unchanged_and_never_calls_llm(ctx):
    game = ctx.game

    use_fixed_menu(ctx, ctx.pw)
    assert game.knows_password is False
    assert game.dialogue == f"[{ih.PASSWORD_NPC_NAME}]: 위치부터 알아오게."

    use_fixed_menu(ctx, ctx.loc)
    assert game.knows_location is True
    assert f"({ctx.x}, {ctx.y})" in game.dialogue
    assert game.objective == "목표: 보물상자의 암호 알아내기"

    use_fixed_menu(ctx, ctx.pw)
    assert game.knows_password is True
    assert f"'{ctx.password}'" in game.dialogue
    assert game.objective == "목표: 보물상자를 열기"

    assert ctx.llm.calls == []
