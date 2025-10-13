import re

import pygame

from game.games.game import Game
from game.games.states import GameState


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
        """Handles the logic for chatting with an NPC."""
        if not self.game.active_npc:
            return

        player_msg = {"role": "user", "content": self.game.input_text}
        self.game.active_npc.chat_history.append(player_msg)

        prompt = self.game.active_npc.chat_history

        response_data = self.game.llm_client.chat(
            prompt, system_prompt=self.game.active_npc.background
        )

        if response_data:
            response = self._parse_llm_response(response_data)

            # Strip NPC name prefix if it exists
            name_prefixes = [
                f"[{self.game.active_npc.name}]:",
                f"{self.game.active_npc.name}:",
            ]
            for prefix in name_prefixes:
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
            response = emoji_pattern.sub(r"", response)

            self._check_info_revelation(response)
        else:
            response = "..."  # Default response on error

        self.game.active_npc.chat_history.append(
            {"role": "assistant", "content": response}
        )
        # --- End of LLM Integration ---

        if len(self.game.active_npc.chat_history) > 6:  # Keep chat history concise
            self.game.active_npc.chat_history.pop(0)
            self.game.active_npc.chat_history.pop(0)

        self.game.input_text = ""
        self._update_chat_display()

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

    def _check_info_revelation(self, response: str) -> None:
        """Checks if the LLM's response reveals critical game information."""
        if not self.game.active_npc:
            return
        if (
            self.game.active_npc.name == "위치 정보원"
            and str(self.game.treasure_pos[0]) in response
        ):
            self.game.knows_location = True
        if (
            self.game.active_npc.name == "암호 전문가"
            and self.game.password in response
        ):
            self.game.knows_password = True

    def _update_chat_display(self) -> None:
        """Formats the active NPC's chat history for display."""
        if not self.game.active_npc:
            self.game.chat_display_text = ""
            return

        self.game.chat_display_text = "\n".join(
            [msg["content"] for msg in self.game.active_npc.chat_history]
        )