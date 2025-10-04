import json
from typing import Any, Optional

import requests


class OllamaClient:
    """
    A client for interacting with the Ollama API.
    """

    def __init__(self, host: str = "localhost", port: int = 11434):
        """
        Initializes the OllamaClient.

        Args:
            host (str): The hostname or IP address of the Ollama server.
            port (int): The port number of the Ollama server.
        """
        self.base_url = f"http://{host}:{port}"

    def chat(
        self, model: str, messages: list[dict[str, str]], stream: bool = False
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
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama server: {e}")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON response from Ollama server.")
            return None


if __name__ == "__main__":
    # Example usage of the OllamaClient

    # 1. Create a client instance
    client = OllamaClient()

    # 2. Define model and prompt
    model_name = "exaone3.5:2.4b"  # Make sure you have this model
    user_prompt = "안녕하세요. 인사좀 해줘요."
    conversation = [
        {
            "role": "system",
            "content": "너는 욕쟁이 할머니야. 걸쭉한 사투리로 대답해줘.",
        },
        {"role": "user", "content": user_prompt},
    ]

    # 3. Generate a response
    print(f"Sending chat to {model_name}: '{user_prompt}'")
    response_data = client.chat(model=model_name, messages=conversation)

    # 4. Print the response
    if response_data:
        print("\nFull response data:")
        print(response_data)
        print("\nGenerated response:")
        # For /api/chat, the response is in response_data['message']['content']
        print(
            response_data.get("message", {}).get("content", "No response text found.")
        )
