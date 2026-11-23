from typing import Optional, Union

import requests

# (connect, read) timeout in seconds for the single non-streaming POST.
# ``requests`` applies the read timeout to each wait for bytes from the
# socket, not the total wall-clock duration. A normal non-streaming Ollama
# reply sends its bytes once generation is finished; a trickling peer can
# still exceed the read-inactivity timeout.
DEFAULT_TIMEOUT: tuple[float, float] = (5.0, 120.0)

Timeout = Union[float, tuple[float, float]]


class OllamaClient:
    """
    A minimal client for the local Ollama ``/api/chat`` endpoint.

    Only **non-streaming** requests are supported. A streaming request returns
    newline-delimited JSON (NDJSON), not one JSON document, and this client
    does not parse it. ``stream=True`` is rejected rather than misread.

    ``chat`` returns the reply text, or ``None`` (the failure sentinel) if the
    server cannot be reached, times out, returns a non-2xx status, or returns a
    body without a non-empty string at ``message.content``. A short reason is
    kept in ``last_error``. Error messages never include the request messages,
    the system prompt or the response body.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model=None,
        timeout: Timeout = DEFAULT_TIMEOUT,
    ):
        """
        Initializes the OllamaClient.

        Args:
            host (str): The hostname or IP address of the Ollama server.
            port (int): The port number of the Ollama server.
            model (str | None): Default model name (e.g. ``"gpt-oss:20b"``).
            timeout (float | tuple[float, float]): ``requests`` timeout for the
                POST, either a single value or ``(connect, read)``. Must not be
                ``None``; the wait is always bounded.
        """
        if timeout is None:
            raise ValueError("timeout must be set; unbounded waits are not allowed")
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.timeout = timeout
        self.last_error: Optional[str] = None

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model=None,
        system_prompt=None,
    ) -> Optional[str]:
        """
        Sends a chat conversation to the Ollama API and waits for one reply.

        Args:
            messages (list[dict[str, str]]): The conversation. A leading system
                message is replaced by ``system_prompt`` (or the default).
            stream (bool): Must be False. Streaming (NDJSON) is not supported.
            model (str | None): Overrides the client's default model.
            system_prompt (str | None): System prompt to send first.

        Returns:
            Optional[str]: The non-empty reply text from ``message.content``,
            or ``None`` on any transport, HTTP, JSON or shape failure.

        Raises:
            ValueError: If ``stream`` is True.
        """
        if stream:
            raise ValueError(
                "OllamaClient.chat supports only stream=False; streaming "
                "responses are NDJSON and are not parsed by this client"
            )
        self.last_error = None

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
            "stream": False,
        }

        try:
            response = requests.post(url, json=data, timeout=self.timeout)
        except requests.exceptions.Timeout:
            return self._fail("request timed out")
        except requests.exceptions.ConnectionError:
            return self._fail("could not connect to the Ollama server")
        except requests.exceptions.RequestException as e:
            return self._fail(f"request failed ({type(e).__name__})")

        status = getattr(response, "status_code", None)
        if not isinstance(status, int) or not 200 <= status < 300:
            return self._fail(f"HTTP status {status}")

        try:
            body = response.json()
        except ValueError:  # includes json / requests JSONDecodeError
            return self._fail("response is not a single JSON document")

        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return self._fail("response has no textual message.content")
        if not content.strip():
            return self._fail("response message.content is empty")
        return content

    def _fail(self, reason: str) -> None:
        """Records a sanitized failure reason and returns the failure sentinel."""
        self.last_error = reason
        print(f"Ollama chat failed: {reason}")
        return None

    def has_system_prompt(self, messages):
        if not messages:
            return False
        to_check = messages[0]
        return isinstance(to_check, dict) and to_check.get("role") == "system"

    def default_system_prompt(self):
        return "You are a helpful assistant."
