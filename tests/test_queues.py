import unittest

import pytest

from aiolibs_executor._executor import QueueShutDown, _WorkerQueue


class TestWorkerQueue(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_get_nowait(self):
        q = _WorkerQueue()
        q.put_nowait(1)

        i = q.get_nowait()
        assert i == 1

        q.put_nowait(2)
        q.shutdown()
        with pytest.raises(QueueShutDown):
            q.get_nowait()

    async def test_shutdown_put_nowait(self):
        q = _WorkerQueue()
        q.put_nowait(1)

        i = q.get_nowait()
        assert i == 1

        q.shutdown()
        with pytest.raises(QueueShutDown):
            q.put_nowait(2)

    async def test_shutdown_with_get_await(self):
        q = _WorkerQueue()
        await q.put(1)

        i = await q.get()
        assert i == 1

        q.shutdown()
        with pytest.raises(QueueShutDown):
            await q.put(2)
