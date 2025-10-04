import json
from typing import Any, Optional

import requests


class OllamaClient:
    """
    A client for interacting with the Ollama API.
    """

    def __init__(self, host: str = "localhost", port: int = 11434, model=None):
        """
        Initializes the OllamaClient.

        Args:
            host (str): The hostname or IP address of the Ollama server.
            port (int): The port number of the Ollama server.
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model=None,
        system_prompt=None,
    ) -> Optional[dict[str, Any]]:
        """
        Sends a chat conversation to the Ollama API.

        Args:
            model (str): The name of the model to use (e.g., 'llama3').
            messages (list[dict[str, str]]): A list of messages in the conversation.
            stream (bool): Whether to stream the response or not. Defaults to False.

        Returns:
            Optional[dict]: The JSON response from the API, or None if an error occurs.
        """
        url = f"{self.base_url}/api/chat"
        new_messages = list(messages)
        if self.has_system_prompt(new_messages):
            new_messages = new_messages[1:]
        new_messages.insert(
            0,
            {
                "role": "system",
                "content": system_prompt or self.default_system_prompt(),
            },
        )

        data = {
            "model": model or self.model,
            "messages": new_messages,
            "stream": stream,
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json().get("message").get("content")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama server: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON response from Ollama server.")
            return None

    def has_system_prompt(self, messages):
        to_check = messages[0]
        if to_check["role"] == "system":
            return True
        return False

    def default_system_prompt(self):
        return "You are a helpful assistant."