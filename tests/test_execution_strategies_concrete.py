import pytest
import time
import asyncio
from autogen_extensions.execution_strategies_concrete import (
    SequentialStrategy,
    ThreadPoolStrategy,
    ProcessPoolStrategy,
    AsyncStrategy,
)

# Dummy tasks for testing


def task_a():
    return "A"


def task_b():
    return "B"


def task_sleep():
    time.sleep(0.1)
    return "slept"


def task_error():
    raise ValueError("fail")


# Async tasks


async def async_task_a():
    await asyncio.sleep(0.01)
    return "A"


async def async_task_b():
    await asyncio.sleep(0.01)
    return "B"


async def async_task_error():
    raise RuntimeError("fail")


def test_sequential_strategy():
    strat = SequentialStrategy()
    assert strat.run_tasks([task_a, task_b]) == ["A", "B"]
    assert strat.run_tasks([task_sleep, task_a]) == ["slept", "A"]
    with pytest.raises(ValueError):
        strat.run_tasks([task_a, task_error, task_b])


def test_threadpool_strategy():
    strat = ThreadPoolStrategy(max_workers=2)
    results = strat.run_tasks([task_a, task_b, task_sleep])
    assert sorted(results) == ["A", "B", "slept"]
    with pytest.raises(ValueError):
        strat.run_tasks([task_a, task_error, task_b])


def test_processpool_strategy():
    strat = ProcessPoolStrategy(max_workers=2)
    results = strat.run_tasks([task_a, task_b, task_sleep])
    assert sorted(results) == ["A", "B", "slept"]
    with pytest.raises(Exception):
        strat.run_tasks([task_a, task_error, task_b])


def test_async_strategy():
    strat = AsyncStrategy()
    results = strat.run_tasks([lambda: async_task_a(), lambda: async_task_b()])
    assert sorted(results) == ["A", "B"]
    with pytest.raises(RuntimeError):
        strat.run_tasks([lambda: async_task_a(), lambda: async_task_error()])
