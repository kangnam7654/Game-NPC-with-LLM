import re

import pygame

from configs import config
from game.games.game import Game
from game.games.states import GameState

LOCATION_NPC_NAME = "위치 정보원"
PASSWORD_NPC_NAME = "암호 전문가"

# Maximum number of user/assistant messages kept per NPC.
MAX_CHAT_HISTORY = 6

# Shown in the info panel when an NPC reply could not be obtained. The
# player's text stays in the input box, so pressing Enter retries it.
CHAT_FAILURE_NOTICE = (
    "NPC의 응답을 받지 못했습니다 (Ollama 연결/응답 오류). "
    "입력한 내용은 그대로 남아 있으니 Enter로 다시 보내거나 ESC로 나가세요."
)

# Used instead of the real password in the password NPC's system prompt until
# the game has recorded that the player knows the treasure location. Contains
# no digits, so the secret is never sent to the model before then.
LOCKED_PASSWORD_PLACEHOLDER = (
    "[봉인됨: 상대가 아직 보물의 위치를 모르므로 지금은 암호를 말할 수 없습니다. "
    "숫자를 지어내지 말고, 먼저 보물의 위치를 알아오라고 하세요]"
)

# --- Revelation heuristic -------------------------------------------------
# This is a bounded text heuristic, not semantic understanding:
# * Location: the location NPC's reply must mention the treasure/location and
#   contain an explicit coordinate pair -- "(x, y)", "[x, y]", "좌표 x, y" or
#   "x=.. y=.." -- and every such pair must be the correct one. A lone x or
#   other incidental numbers never count.
# * Password: only the password NPC, only after the location is known, and only
#   for an explicit statement like "암호는 '7'" / "password is 7" (keyword
#   followed within a few characters by a number). All such statements must
#   give the correct digit.
# * A pair/statement is ignored if its sentence looks negated
#   ("아니", "없", "않", "모르", "못", "not", ...) or ends with "?".
# * Values the player typed in the same message are ignored, so echoing or
#   confirming the player's own guess does not grant progress.
# Spelled-out numbers, unusual phrasings or a misleading model reply can still
# be misjudged; the fixed-information menu is the reliable path.
_LOCATION_KEYWORDS = ("보물", "위치", "좌표", "treasure", "location", "coordinate")
_COORD_PAIR_PATTERNS = (
    re.compile(r"[(\[]\s*(\d+)\s*,\s*(\d+)\s*[)\]]"),
    re.compile(r"좌표\s*(?:는|은|가|:)?\s*(\d+)\s*,\s*(\d+)(?!\d)"),
    re.compile(
        r"(?<![a-z])x\s*(?:좌표)?\s*(?:는|은|가|=|:)?\s*(\d+)"
        r"[^\d\n]{0,6}?(?<![a-z])y\s*(?:좌표)?\s*(?:는|은|가|=|:)?\s*(\d+)",
        re.IGNORECASE,
    ),
)
_BARE_PAIR_PATTERN = re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)(?!\d)")
_PASSWORD_STATEMENT_PATTERN = re.compile(
    r"(?:암호|비밀번호|비번|password|passcode)[^\d\n.!?]{0,12}?(?<!\d)(\d+)(?!\d)",
    re.IGNORECASE,
)
_NEGATION_TOKENS = (
    "아니", "아닐", "아닌", "아님", "아냐", "없", "않", "모르", "못",
    "not", "n't", "never",
)
_SENTENCE_DELIMITERS = ".!?\n。"


def _sentence_around(text: str, start: int, end: int) -> tuple[str, str]:
    """Returns the sentence containing ``text[start:end]`` and its end mark."""
    left = max(text.rfind(d, 0, start) for d in _SENTENCE_DELIMITERS) + 1
    right = len(text)
    for i in range(end, len(text)):
        if text[i] in _SENTENCE_DELIMITERS:
            right = i
            break
    terminator = text[right] if right < len(text) else ""
    return text[left:right], terminator


def _is_assertion(text: str, match: re.Match) -> bool:
    """True unless the match sits in a question or a negated sentence."""
    sentence, terminator = _sentence_around(text, match.start(), match.end())
    if terminator == "?":
        return False
    lowered = sentence.lower()
    return not any(token in lowered for token in _NEGATION_TOKENS)


class InteractionHandler:
    """Handles player interactions with NPCs, objects, and text input."""

    def __init__(self, game: Game):
        """Initializes the InteractionHandler.

        Args:
            game (Game): The main game object.
        """
        self.game = game

    def handle_interaction(self) -> None:
        """Handles player interaction with NPCs and the treasure chest."""
        for npc in self.game.npcs:
            if self.game.is_adjacent(self.game.player_pos, npc.pos):
                self.game.state = GameState.INTERACTION_MENU
                self.game.active_npc = npc
                self.game.menu_selection = 0
                return
        
        

        if (
            self.game.treasure_visible
            and self.game.player_pos == self.game.treasure_pos
            and not self.game.treasure_opened
        ):
            if self.game.knows_password:
                pygame.key.start_text_input()
                self.game.state = GameState.TEXT_INPUT
                self.game.input_prompt = "암호를 입력하세요 (Enter로 확인):"
            else:
                self.game.dialogue = "상자가 잠겨있다. 암호를 알아내야 한다."
            return

        self.game.dialogue = "주변에 상호작용할 대상이 없습니다."

    def process_text_input(self) -> None:
        """Processes text input from the user for NPC chat or password entry."""
        if self.game.active_npc:
            self._process_npc_chat()
        elif self.game.input_prompt:
            self._process_password_entry()

    def _process_npc_chat(self) -> None:
        """Handles the logic for chatting with an NPC.

        On success the player message and the cleaned reply are appended to the
        NPC's history (bounded to ``MAX_CHAT_HISTORY``), revelation is checked
        and the input box is cleared. On any failure (no client, exception,
        ``None``/non-text/empty reply) nothing is added to the history, no
        quest flag changes, the input text is kept for a retry and
        ``CHAT_FAILURE_NOTICE`` is shown.
        """
        npc = self.game.active_npc
        if not npc:
            return

        player_text = self.game.input_text
        if not player_text.strip():
            return  # Nothing to send; don't call the LLM with an empty message.

        prompt = list(npc.chat_history) + [{"role": "user", "content": player_text}]

        response = ""
        client = self.game.llm_client
        if client is not None:
            try:
                response_data = client.chat(
                    prompt, system_prompt=self._system_prompt_for(npc)
                )
            except Exception as e:  # keep the game loop alive
                print(f"NPC chat failed ({type(e).__name__})")
                response_data = None
            response = self._clean_response(response_data, npc.name)

        if not response:
            self.game.dialogue = CHAT_FAILURE_NOTICE
            self._update_chat_display()
            return

        if self.game.dialogue == CHAT_FAILURE_NOTICE:
            self.game.dialogue = ""

        self._check_info_revelation(response, player_text)

        npc.chat_history.append({"role": "user", "content": player_text})
        npc.chat_history.append({"role": "assistant", "content": response})
        del npc.chat_history[:-MAX_CHAT_HISTORY]

        self.game.input_text = ""
        self._update_chat_display()

    def _system_prompt_for(self, npc) -> str:
        """Returns the NPC's system prompt, hiding the password until location is known."""
        if npc.name == PASSWORD_NPC_NAME and not self.game.knows_location:
            with open(config.NPC_PW_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read().format(LOCKED_PASSWORD_PLACEHOLDER)
        return npc.background

    def _clean_response(self, response_data, npc_name: str) -> str:
        """Turns raw client output into display text; returns "" if unusable."""
        if not isinstance(response_data, str):
            return ""
        response = self._parse_llm_response(response_data)

        # Strip NPC name prefix if it exists
        for prefix in (f"[{npc_name}]:", f"{npc_name}:"):
            if response.startswith(prefix):
                response = response[len(prefix) :].lstrip()
                break

        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\u2600-\u26ff"  # miscellaneous symbols
            "\u2700-\u27bf"  # dingbats
            "\u3000-\u303f"  # CJK Symbols and Punctuation
            "\ufe0f"  # variation selector
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", response).strip()

    def _process_password_entry(self) -> None:
        """Handles the logic for entering the treasure password."""
        if self.game.input_text == self.game.password:
            self.game.treasure_opened = True
            self.game.dialogue = (
                f"암호 '{self.game.password}'를 입력했다. 보물상자가 열렸다!"
            )
            self.game.objective = "목표: 출구로 탈출하기"
        else:
            self.game.dialogue = "암호가 틀렸습니다."

        self.game.input_text = ""
        pygame.key.stop_text_input()
        self.game.state = GameState.PLAYING
        self.game.input_prompt = ""
        self.game.active_npc = None

    def _parse_llm_response(self, response_data: str) -> str:
        """Parses the raw response from the LLM to extract clean text."""
        response = response_data.split("</think>")[-1]
        if "[|assistant|]" in response:
            response = response.split("[|assistant|]", 1)[1]
        return response.replace("[|endofturn|]", "").strip()

    def _check_info_revelation(self, response: str, player_text: str = "") -> None:
        """Checks if the active NPC's reply explicitly reveals quest information.

        Uses the bounded heuristic documented at the top of this module. Only
        the active NPC's own information can be granted, and the password only
        once the location is known.
        """
        npc = self.game.active_npc
        if not npc or not isinstance(response, str):
            return
        if npc.name == LOCATION_NPC_NAME and self._reveals_location(
            response, player_text
        ):
            self.game.knows_location = True
        elif (
            npc.name == PASSWORD_NPC_NAME
            and self.game.knows_location
            and self._reveals_password(response, player_text)
        ):
            self.game.knows_password = True

    def _reveals_location(self, response: str, player_text: str) -> bool:
        lowered = response.lower()
        if not any(keyword in lowered for keyword in _LOCATION_KEYWORDS):
            return False
        echoed = self._coordinate_pairs(player_text, assertions_only=False)
        echoed |= {
            (int(x), int(y)) for x, y in _BARE_PAIR_PATTERN.findall(player_text)
        }
        asserted = self._coordinate_pairs(response, assertions_only=True)
        target = (int(self.game.treasure_pos[0]), int(self.game.treasure_pos[1]))
        # Reject contradictory coordinates even when one was supplied by the
        # player. Subtract only for deciding whether *new* information appeared.
        return bool(asserted) and asserted == {target} and target not in echoed

    @staticmethod
    def _coordinate_pairs(text: str, assertions_only: bool) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for pattern in _COORD_PAIR_PATTERNS:
            for match in pattern.finditer(text):
                if assertions_only and not _is_assertion(text, match):
                    continue
                pairs.add((int(match.group(1)), int(match.group(2))))
        return pairs

    def _reveals_password(self, response: str, player_text: str) -> bool:
        password = str(self.game.password)
        if re.search(rf"(?<!\d){re.escape(password)}(?!\d)", player_text):
            return False  # the player supplied this value; don't grant on an echo
        stated = {
            match.group(1)
            for match in _PASSWORD_STATEMENT_PATTERN.finditer(response)
            if _is_assertion(response, match)
        }
        return stated == {password}

    def _update_chat_display(self) -> None:
        """Formats the active NPC's chat history for display."""
        if not self.game.active_npc:
            self.game.chat_display_text = ""
            return

        self.game.chat_display_text = "\n".join(
            [msg["content"] for msg in self.game.active_npc.chat_history]
        )