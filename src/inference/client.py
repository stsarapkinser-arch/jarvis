"""Client for the standalone inference server.

Replaces direct llama-cpp usage in orchestrator.py.
Handles server crashes gracefully by detecting disconnects and restarting.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .protocol import GenerateRequest, GenerateChunk, HealthCheck, HealthReply, ServerError

log = logging.getLogger("jarvis.llm_client")


class InferenceClient:
    """Async client to the inference server.

    Automatically starts the server if it's not running.
    """

    def __init__(
        self,
        socket_path: str | None = None,
        model_path: str | None = None,
    ):
        self.socket_path = socket_path or os.getenv(
            "JARVIS_LLAMA_SOCKET",
            "/tmp/jarvis-llm.sock"
        )
        self.model_path = model_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._connect_lock = asyncio.Lock()

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream tokens from the model.

        Yields tokens one by one. If the server crashes, raises an exception
        which the caller (orchestrator) should catch and handle gracefully."""
        req = GenerateRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        await self._ensure_connected()

        try:
            # Send request as JSON line
            req_line = json.dumps({"_type": "generate", **req._asdict()})
            self._writer.write(req_line.encode() + b"\n")  # type: ignore[union-attr]
            await self._writer.drain()  # type: ignore[union-attr]

            # Stream responses
            while True:
                line = await self._reader.readline()  # type: ignore[union-attr]
                if not line:
                    log.error("Server disconnected unexpectedly")
                    self._disconnect()
                    raise RuntimeError("Inference server disconnected")

                try:
                    resp_data = json.loads(line.decode())

                    # Check for error
                    if "error" in resp_data:
                        err = ServerError(**resp_data)
                        log.error("Server error: %s — %s", err.error, err.detail)
                        raise RuntimeError(f"Server error: {err.error}")

                    # Check for chunk
                    if "token" in resp_data:
                        chunk = GenerateChunk(**resp_data)
                        if chunk.is_done:
                            break
                        yield chunk.token

                except json.JSONDecodeError:
                    log.error("Malformed response from server: %r", line)
                    self._disconnect()
                    raise RuntimeError("Server protocol error")

        except Exception:
            self._disconnect()
            raise

    async def health_check(self) -> bool:
        """Check if the server is alive."""
        try:
            await self._ensure_connected()
            req_line = json.dumps({"_type": "health_check"})
            self._writer.write(req_line.encode() + b"\n")  # type: ignore[union-attr]
            await self._writer.drain()  # type: ignore[union-attr]

            line = await asyncio.wait_for(
                self._reader.readline(),  # type: ignore[union-attr]
                timeout=2.0,
            )
            if not line:
                return False

            resp_data = json.loads(line.decode())
            return resp_data.get("status") == "ok"

        except Exception as e:
            log.debug("Health check failed: %s", e)
            self._disconnect()
            return False

    async def _ensure_connected(self) -> None:
        """Connect to the server, starting it if necessary."""
        async with self._connect_lock:
            if self._reader is not None and self._writer is not None:
                # Already connected; do a quick health check
                if not await self._health_check_quick():
                    self._disconnect()
                else:
                    return

            # Try to connect; if that fails, start the server and retry
            for attempt in range(2):
                try:
                    self._reader, self._writer = await asyncio.open_connection(
                        path=self.socket_path
                    )
                    log.info("Connected to inference server")
                    return
                except (FileNotFoundError, ConnectionRefusedError):
                    if attempt == 0:
                        log.warning("Server not running; starting it...")
                        await self._start_server()
                    else:
                        raise RuntimeError("Could not connect to inference server")

    async def _health_check_quick(self) -> bool:
        """Quick health check without raising on failure."""
        try:
            req_line = json.dumps({"_type": "health_check"})
            self._writer.write(req_line.encode() + b"\n")  # type: ignore[union-attr]
            await self._writer.drain()  # type: ignore[union-attr]

            line = await asyncio.wait_for(
                self._reader.readline(),  # type: ignore[union-attr]
                timeout=0.5,
            )
            if not line:
                return False
            resp_data = json.loads(line.decode())
            return resp_data.get("status") == "ok"

        except Exception:
            return False

    async def _start_server(self) -> None:
        """Start the inference server as a subprocess."""
        try:
            # Run the server in a subprocess
            env = os.environ.copy()
            if self.model_path:
                env["JARVIS_LLAMA_MODEL_PATH"] = self.model_path
            env["JARVIS_LLAMA_SOCKET"] = self.socket_path

            # Find the module path
            module_path = str(Path(__file__).resolve().parent)

            self._server_proc = subprocess.Popen(
                [sys.executable, "-m", "src.inference.server"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Separate process group so Ctrl+C doesn't kill it
            )
            log.info("Started inference server (PID %d)", self._server_proc.pid)

            # Give it a moment to start
            await asyncio.sleep(0.5)

        except Exception as e:
            log.error("Failed to start inference server: %s", e)
            raise

    def _disconnect(self) -> None:
        """Disconnect from the server."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        log.debug("Disconnected from server")

    def __del__(self) -> None:
        """Clean up on deletion."""
        self._disconnect()
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=2)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
