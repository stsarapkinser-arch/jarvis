"""Standalone inference server for llama-cpp.

Runs in a separate process. If it crashes, Jarvis can restart it without
being killed. Communicates with the client via a Unix socket.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    Llama = None  # type: ignore
    _HAS_LLAMA_CPP = False

from .protocol import GenerateRequest, GenerateChunk, HealthReply, ServerError

log = logging.getLogger("jarvis.llm_server")

# Inline script для subprocess-пробы GPU. Запускается в изолированном процессе,
# чтобы перехватить C-level падение (SIGABRT/SIGSEGV от Vulkan/GGML_ASSERT)
# до того как оно убьёт основной сервер.
# Аргументы: model_path n_gpu_layers n_ctx n_batch n_ubatch
_GPU_PROBE_SCRIPT = """\
import sys
from llama_cpp import Llama
Llama(
    model_path=sys.argv[1],
    n_gpu_layers=int(sys.argv[2]),
    n_ctx=int(sys.argv[3]),
    n_batch=int(sys.argv[4]),
    n_ubatch=int(sys.argv[5]),
    n_threads=1,
    flash_attn=False,
    verbose=False,
)
"""


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
        """Start the server listening on the Unix domain socket.

        Используем ``asyncio.start_unix_server`` — НЕ ``start_server(path=...)``.
        В Python 3.13 ``start_server`` пробрасывает ``path`` в
        ``loop.create_server()``, который такого аргумента не принимает →
        ``TypeError: create_server() got an unexpected keyword argument 'path'``
        (именно этот краш зацикливал сервер инференса в логах оператора).
        ``start_unix_server`` — штатный путь для AF_UNIX и сразу даёт
        (reader, writer) в колбэк."""
        if Path(self.socket_path).exists():
            Path(self.socket_path).unlink()

        self._server = await asyncio.start_unix_server(
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

    async def _probe_gpu(
        self, n_gpu: int, n_ctx: int, n_batch: int, n_ubatch: int
    ) -> bool:
        """Load the model in an isolated subprocess to detect C-level GPU crashes.

        SIGABRT/SIGSEGV from Vulkan или GGML_ASSERT нельзя поймать через
        try/except в Python — они убивают процесс. Проба запускает загрузку в
        дочернем процессе: если тот крашится (returncode != 0), основной сервер
        выживает и может откатиться на CPU.

        Установите JARVIS_LLAMA_GPU_PROBE=0, чтобы пропустить пробу."""
        if os.getenv("JARVIS_LLAMA_GPU_PROBE", "1") == "0":
            return True
        log.info(
            "GPU probe: loading model with n_gpu_layers=%d n_ctx=%d (subprocess test)...",
            n_gpu, n_ctx,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _GPU_PROBE_SCRIPT,
                self.model_path, str(n_gpu), str(n_ctx), str(n_batch), str(n_ubatch),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                log.warning("GPU probe timed out after 120 s — assuming GPU unavailable")
                return False
            if proc.returncode != 0:
                log.warning(
                    "GPU probe exited %d — GPU model load failed (SIGABRT=%s SIGSEGV=%s)",
                    proc.returncode,
                    proc.returncode == -6,   # SIGABRT
                    proc.returncode == -11,  # SIGSEGV
                )
                return False
            log.info("GPU probe succeeded — proceeding with iGPU")
            return True
        except Exception as e:
            log.warning("GPU probe error: %s", e)
            return False

    async def _ensure_llama(self) -> bool:
        """Load the model if not already loaded. Probes GPU before real load."""
        if self._llama is not None:
            return True

        if not _HAS_LLAMA_CPP:
            log.error("llama-cpp-python not installed")
            return False

        if not Path(self.model_path).is_file():
            log.error("Model not found: %s", self.model_path)
            return False

        n_gpu = _env_int("JARVIS_LLAMA_GPU_LAYERS", -1)
        n_ctx = _env_int("JARVIS_LLAMA_CTX", 4096)
        n_batch = _env_int("JARVIS_LLAMA_BATCH", 256)
        n_ubatch = _env_int("JARVIS_LLAMA_UBATCH", 128)
        n_threads = _env_int("JARVIS_LLAMA_THREADS", 4)

        if n_gpu != 0:
            if not await self._probe_gpu(n_gpu, n_ctx, n_batch, n_ubatch):
                log.warning(
                    "GPU probe failed — falling back to CPU-only (n_gpu_layers=0). "
                    "Set JARVIS_LLAMA_GPU_LAYERS=0 to suppress this probe."
                )
                n_gpu = 0

        try:
            log.info(
                "Loading %s (gpu_layers=%d ctx=%d threads=%d batch=%d ubatch=%d)",
                Path(self.model_path).name, n_gpu, n_ctx, n_threads, n_batch, n_ubatch,
            )
            self._llama = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu,
                n_threads=n_threads,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                flash_attn=False,
                chat_format="chatml",
                verbose=False,
            )
            log.info("Model loaded successfully (gpu_layers=%d)", n_gpu)
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
