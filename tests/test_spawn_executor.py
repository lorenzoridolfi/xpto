import pytest
import time
from autogen_extensions.spawn_executor import SpawnExecutor
from autogen_extensions.execution_strategies_concrete import (
    SequentialStrategy,
    ThreadPoolStrategy,
    ProcessPoolStrategy,
    AsyncStrategy,
)
import asyncio

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


def test_spawn_executor_sequential():
    executor = SpawnExecutor(SequentialStrategy())
    assert executor.run([task_a, task_b]) == ["A", "B"]
    assert executor.run([task_sleep, task_a]) == ["slept", "A"]
    with pytest.raises(ValueError):
        executor.run([task_a, task_error, task_b])


def test_spawn_executor_threadpool():
    executor = SpawnExecutor(ThreadPoolStrategy(max_workers=2))
    results = executor.run([task_a, task_b, task_sleep])
    assert sorted(results) == ["A", "B", "slept"]
    with pytest.raises(ValueError):
        executor.run([task_a, task_error, task_b])


def test_spawn_executor_processpool():
    executor = SpawnExecutor(ProcessPoolStrategy(max_workers=2))
    results = executor.run([task_a, task_b, task_sleep])
    assert sorted(results) == ["A", "B", "slept"]
    with pytest.raises(Exception):
        executor.run([task_a, task_error, task_b])


def test_spawn_executor_async():
    executor = SpawnExecutor(AsyncStrategy())
    results = executor.run([lambda: async_task_a(), lambda: async_task_b()])
    assert sorted(results) == ["A", "B"]
    with pytest.raises(RuntimeError):
        executor.run([lambda: async_task_a(), lambda: async_task_error()])
