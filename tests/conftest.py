"""Общая конфигурация pytest для Jarvis.

Зачем этот файл: в проекте почти все async-тесты написаны как обычные
``def test_...():`` с ``asyncio.run(...)`` внутри — это работает без сторонних
плагинов. Но ``tests/test_pixel_bridge.py`` использует другой стиль:
``@pytest.mark.asyncio`` + ``async def test_...()``, который требует пакета
``pytest-asyncio``. Этого пакета в окружении может не быть → 8 тестов падали с
«async def functions are not natively supported».

Чтобы не плодить зависимость и не переписывать тесты, добавляем крошечный
адаптер: любой ``async def`` тест запускается через ``asyncio.run`` прямо в
``pytest_pyfunc_call``. Работает одинаково и с pytest-asyncio, и без него.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Регистрируем маркер, чтобы не было предупреждений про unknown mark
    # (и чтобы --strict-markers не падал).
    config.addinivalue_line(
        "markers",
        "asyncio: запускать async-тест через asyncio.run (адаптер без pytest-asyncio)",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    """Если тест — корутина, выполняем её через asyncio.run.

    Возвращаем True, сигнализируя pytest, что вызов теста уже обработан.
    Для обычных (синхронных) тестов возвращаем None — pytest обрабатывает их
    штатно."""
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        # Прокидываем только те fixture-аргументы, что объявлены в сигнатуре.
        argnames = pyfuncitem._fixtureinfo.argnames
        kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
        asyncio.run(test_func(**kwargs))
        return True
    return None
