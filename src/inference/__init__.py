# Inference layer — общение с нативным llama-server (OpenAI-совместимый REST).
#
# Веса модели живут в ОТДЕЛЬНОМ системном процессе (jarvis-llm.service), что
# изолирует Jarvis от нативных крашей бэкенда и не блокирует event-loop.
#
#   openai_client.LlamaServerClient — async httpx-клиент к /v1/chat/completions
#   tools                           — JSON-схемы инструментов (Function Calling)
#   router.IntentRouter             — семантический маршрутизатор + микро-промпты
#   agent.run_agent                 — обобщённый tool-calling цикл
