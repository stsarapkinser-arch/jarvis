"""Standalone inference server for llama-cpp.

Runs in a separate process. If it crashes, Jarvis can restart it without
being killed. Communicates with the client via a Unix socket or TCP.

Usage:
    python -m src.inference.server --socket /tmp/jarvis-llm.sock

Environment variables:
    JARVIS_LLAMA_MODEL_PATH    - Path to GGUF model
    JARVIS_LLAMA_GPU_LAYERS    - Number of layers to offload to GPU (-1 = all)
    JARVIS_LLAMA_THREADS       - Number of CPU threads for inference
    JARVIS_LLAMA_CTX           - Context window size
    JARVIS_LLAMA_BATCH         - Batch size
    JARVIS_LLAMA_UBATCH        - Micro-batch size
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    Llama = None  # type: ignore
    _HAS_LLAMA_CPP = False

from .protocol import GenerateRequest, GenerateChunk, HealthCheck, HealthReply, ServerError

log = logging.getLogger("jarvis.llm_server")


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("env %s=%r is not int — using default %d", name, raw, default)
        return default


class LlamaServer:
    """Runs inference in a protected subprocess."""

    def __init__(self, socket_path: str, model_path: str | None = None):
        self.socket_path = socket_path
        self.model_path = model_path or os.getenv(
            "JARVIS_LLAMA_MODEL_PATH",
            str(Path(__file__).resolve().parents[2] / "models" / "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
        )
        self._llama: Any | None = None
        self._server = None

    async def start(self) -> None:
        """Start the server listening on the socket."""
        if Path(self.socket_path).exists():
            Path(self.socket_path).unlink()

        self._server = await asyncio.start_server(
            self._handle_client,
            path=self.socket_path,
        )
        log.info("Inference server listening on %s", self.socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        addr = writer.get_extra_info("peername")
        log.debug("Client connected from %s", addr)
        try:
            while True:
                # Read one JSON line
                line = await reader.readline()
                if not line:
                    break

                try:
                    req_data = json.loads(line.decode())
                    msg_type = req_data.get("_type")

                    if msg_type == "health_check":
                        reply = HealthReply()
                        writer.write(json.dumps(reply._asdict()).encode() + b"\n")

                    elif msg_type == "generate":
                        await self._handle_generate(
                            GenerateRequest(**{k: v for k, v in req_data.items() if k != "_type"}),
                            writer,
                        )

                    else:
                        err = ServerError(error="unknown_request_type", detail=msg_type)
                        writer.write(json.dumps(err._asdict()).encode() + b"\n")

                except json.JSONDecodeError as e:
                    err = ServerError(error="json_decode_error", detail=str(e))
                    writer.write(json.dumps(err._asdict()).encode() + b"\n")
                except Exception as e:
                    log.exception("Error processing request")
                    err = ServerError(error="server_error", detail=str(e))
                    writer.write(json.dumps(err._asdict()).encode() + b"\n")

                await writer.drain()

        except Exception:
            log.exception("Client handler crashed")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_generate(
        self,
        req: GenerateRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Stream generate tokens to the client."""
        if not await self._ensure_llama():
            err = ServerError(error="model_not_loaded", detail="GGUF not available")
            writer.write(json.dumps(err._asdict()).encode() + b"\n")
            await writer.drain()
            return

        try:
            messages = []
            if req.system:
                messages.append({"role": "system", "content": req.system})
            messages.append({"role": "user", "content": req.prompt})

            # This is the dangerous part: if the model or llama.cpp crashes here,
            # the whole server process dies. But that's OK — the client detects
            # the disconnect and can restart us.
            stream = self._llama.create_chat_completion(  # type: ignore[union-attr]
                messages=messages,
                stream=True,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=1.0,
            )

            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
                content = delta.get("content") if isinstance(delta, dict) else None
                if content:
                    reply = GenerateChunk(token=content, is_done=False)
                    writer.write(json.dumps(reply._asdict()).encode() + b"\n")
                    await writer.drain()

            # Send final "done" marker
            reply = GenerateChunk(token="", is_done=True)
            writer.write(json.dumps(reply._asdict()).encode() + b"\n")
            await writer.drain()

        except Exception as e:
            log.exception("Generation failed")
            err = ServerError(error="generation_failed", detail=str(e))
            writer.write(json.dumps(err._asdict()).encode() + b"\n")
            await writer.drain()

    async def _ensure_llama(self) -> bool:
        """Load the model if not already loaded."""
        if self._llama is not None:
            return True

        if not _HAS_LLAMA_CPP:
            log.error("llama-cpp-python not installed")
            return False

        if not Path(self.model_path).is_file():
            log.error("Model not found: %s", self.model_path)
            return False

        try:
            log.info(
                "Loading %s (gpu_layers=%d threads=%d ctx=%d batch=%d ubatch=%d)",
                Path(self.model_path).name,
                _env_int("JARVIS_LLAMA_GPU_LAYERS", -1),
                _env_int("JARVIS_LLAMA_THREADS", 4),
                _env_int("JARVIS_LLAMA_CTX", 8192),
                _env_int("JARVIS_LLAMA_BATCH", 256),
                _env_int("JARVIS_LLAMA_UBATCH", 128),
            )
            self._llama = Llama(
                model_path=self.model_path,
                n_gpu_layers=_env_int("JARVIS_LLAMA_GPU_LAYERS", -1),
                n_threads=_env_int("JARVIS_LLAMA_THREADS", 4),
                n_ctx=_env_int("JARVIS_LLAMA_CTX", 8192),
                n_batch=_env_int("JARVIS_LLAMA_BATCH", 256),
                n_ubatch=_env_int("JARVIS_LLAMA_UBATCH", 128),
                flash_attn=False,
                chat_format="chatml",
                verbose=False,
            )
            log.info("Model loaded successfully")
            return True
        except Exception:
            log.exception("Model load failed")
            return False

    async def run(self) -> None:
        """Start the server and run until interrupted."""
        await self.start()
        try:
            async with self._server:
                await self._server.serve_forever()
        except KeyboardInterrupt:
            log.info("Server shutdown requested")
        except Exception:
            log.exception("Server crashed")
            sys.exit(1)


async def main() -> None:
    """Entry point for the inference server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    socket_path = os.getenv("JARVIS_LLAMA_SOCKET", "/tmp/jarvis-llm.sock")
    server = LlamaServer(socket_path)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
