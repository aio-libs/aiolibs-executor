import asyncio
import unittest
import warnings
from collections.abc import AsyncIterator
from contextvars import ContextVar, copy_context
from typing import Any

from aiolibs_executor import Executor
from aiolibs_executor._executor import _RateLimiter


class BaseTestCase(unittest.IsolatedAsyncioTestCase):
    def make_executor(
        self,
        num_workers: int = 0,
        *,
        max_pending: int = 0,
        max_throughput: int = 0,
        task_name_prefix: str = "",
    ) -> Executor:
        executor = Executor(
            num_workers=num_workers,
            max_pending=max_pending,
            max_throughput=max_throughput,
            task_name_prefix=task_name_prefix,
        )
        self.addAsyncCleanup(executor.shutdown)
        return executor


class TestSubmit(BaseTestCase):
    async def test_submit_nowait(self) -> None:
        executor = self.make_executor()

        async def f(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            await asyncio.sleep(0)
            return args, kwargs

        self.assertEqual(
            await executor.submit_nowait(f(1, a=2)), ((1,), {"a": 2})
        )

    async def test_submit(self) -> None:
        executor = self.make_executor()

        async def f(
            *args: Any, **kwargs: Any
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            await asyncio.sleep(0)
            return args, kwargs

        fut = await executor.submit(f(1, a=2))
        self.assertEqual(await fut, ((1,), {"a": 2}))

    async def test_map(self) -> None:
        executor = self.make_executor()

        async def f(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        arg = list(range(3))
        ret = [i async for i in executor.map(f, arg, arg)]
        self.assertEqual(ret, [0, 2, 4])

    async def test_amap(self) -> None:
        executor = self.make_executor()

        async def f(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        async def inp() -> AsyncIterator[int]:
            for i in range(1, 4):
                await asyncio.sleep(0)
                yield i

        ret = [i async for i in executor.amap(f, inp(), inp())]
        self.assertEqual(ret, [2, 4, 6])

    async def test_submit_nowait_default_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        c.set(1)

        fut = executor.submit_nowait(f(1))
        self.assertEqual(await fut, 2)

    async def test_submit_nowait_with_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        token = c.set(1)
        context = copy_context()
        c.reset(token)

        fut = executor.submit_nowait(f(1), context=context)
        self.assertEqual(await fut, 2)

    async def test_submit_default_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        c.set(1)

        fut = await executor.submit(f(1))
        self.assertEqual(await fut, 2)

    async def test_submit_with_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        token = c.set(1)
        context = copy_context()
        c.reset(token)

        fut = await executor.submit(f(1), context=context)
        self.assertEqual(await fut, 2)

    async def test_map_default_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        c.set(1)

        ret = [i async for i in executor.map(f, range(3))]
        self.assertEqual(ret, [1, 2, 3])

    async def test_map_with_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        token = c.set(1)
        context = copy_context()
        c.reset(token)

        ret = [i async for i in executor.map(f, range(3), context=context)]
        self.assertEqual(ret, [1, 2, 3])

    async def test_amap_default_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        c.set(1)

        async def inp() -> AsyncIterator[int]:
            for i in range(3):
                await asyncio.sleep(0)
                yield i

        ret = [i async for i in executor.amap(f, inp())]
        self.assertEqual(ret, [1, 2, 3])

    async def test_amap_with_context(self) -> None:
        executor = self.make_executor()

        c: ContextVar[int] = ContextVar("c")

        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + c.get()

        token = c.set(1)
        context = copy_context()
        c.reset(token)

        async def inp() -> AsyncIterator[int]:
            for i in range(3):
                await asyncio.sleep(0)
                yield i

        ret = [i async for i in executor.amap(f, inp(), context=context)]
        self.assertEqual(ret, [1, 2, 3])

    async def test_context_manager(self) -> None:
        async def f(a: int) -> int:
            await asyncio.sleep(0)
            return a + 1

        async with self.make_executor() as executor:
            fut = await executor.submit(f(1))
            self.assertEqual(await fut, 2)


class TestInit(BaseTestCase):
    def test_invalid_num_workers(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "num_workers must be greater than 0"
        ):
            self.make_executor(num_workers=-1)

    def test_invalid_max_pending(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_pending must be non-negative number"
        ):
            self.make_executor(max_pending=-1)

    def test_invalid_max_throughput(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_throughput must be non-negative number"
        ):
            self.make_executor(max_throughput=-1)

    async def test_double_lazy_init(self) -> None:
        executor = self.make_executor()
        loop = executor._lazy_init()[0]
        self.assertIs(loop, asyncio.get_running_loop())
        loop = executor._lazy_init()[0]
        self.assertIs(loop, asyncio.get_running_loop())

    async def test_lazy_init_after_shutdown(self) -> None:
        executor = self.make_executor()
        await executor.shutdown()
        with self.assertRaisesRegex(
            RuntimeError, "cannot schedule new futures after shutdown"
        ):
            executor._lazy_init()

    async def test_lazy_init_from_nonasyncio_if_inited(self) -> None:
        executor = self.make_executor()
        executor._lazy_init()

        def f() -> asyncio.AbstractEventLoop:
            return executor._lazy_init()[0]

        self.assertEqual(
            await asyncio.to_thread(f), asyncio.get_running_loop()
        )

    async def test_lazy_init_from_nonasyncio_if_not_inited(self) -> None:
        executor = self.make_executor()

        def f() -> None:
            executor._lazy_init()

        with self.assertRaisesRegex(RuntimeError, "no running event loop"):
            await asyncio.to_thread(f)

    async def test_lazy_init_bound_to_different_loop(self) -> None:
        executor = self.make_executor()
        executor._lazy_init()

        async def g() -> None:
            executor._lazy_init()

        def f() -> None:
            with asyncio.Runner() as runner:
                runner.run(g())

        with self.assertRaisesRegex(
            RuntimeError, "is bound to a different event loop"
        ):
            await asyncio.to_thread(f)

    async def test_incomplete_initialization(self) -> None:
        executor = self.make_executor()
        executor._lazy_init()
        executor._rate_limiter = None
        with self.assertRaisesRegex(RuntimeError, "fully initialize"):
            executor._lazy_init()


class TestThroughput(BaseTestCase):
    async def test_no_throughput_limit(self) -> None:
        task_count = 512

        executor = self.make_executor(num_workers=128)

        async def f() -> None:
            pass

        start_time = asyncio.get_running_loop().time()
        tasks = [await executor.submit(f()) for _ in range(task_count)]
        await asyncio.gather(*tasks)
        end_time = asyncio.get_running_loop().time()
        self.assertGreater(task_count / (end_time - start_time), 128)

    async def test_throughput(self) -> None:
        executor = self.make_executor(1, max_throughput=1)

        async def f() -> None:
            pass

        task = await executor.submit(f())
        await task

        with warnings.catch_warnings():
            with self.assertRaises(asyncio.TimeoutError):
                async with asyncio.timeout(0.01):
                    task = await executor.submit(f())
                    await task

    async def test_fast_tasks_throttling(self) -> None:
        executor = self.make_executor(128, max_throughput=2)

        async def f(_: int) -> None:
            pass

        start_time = asyncio.get_running_loop().time()
        async for _ in executor.map(f, range(3)):
            pass
        end_time = asyncio.get_running_loop().time()
        throughput = 3 / (end_time - start_time)
        self.assertGreaterEqual(throughput, 2)
        self.assertLess(throughput, 3)

    async def test_slow_tasks_throttling(self) -> None:
        executor = self.make_executor(2, max_throughput=128)

        async def f(_: int) -> None:
            await asyncio.sleep(0.1)

        start_time = asyncio.get_running_loop().time()
        async for _ in executor.map(f, range(10)):
            pass
        end_time = asyncio.get_running_loop().time()
        throughput = 10 / (end_time - start_time)
        # Almost 20 tasks should complete per second (2 workers, 0.1 sec per
        # task)
        self.assertGreater(throughput, 18)
        self.assertLess(throughput, 20)


class TestShutdown(BaseTestCase):
    async def test_shutdown_not_inited(self) -> None:
        executor = self.make_executor()
        await executor.shutdown()

    async def test_shutdown_twice(self) -> None:
        executor = self.make_executor()
        await executor.shutdown()
        await executor.shutdown()

    async def test_shutdown_cancel_futures(self) -> None:
        executor = self.make_executor(1)
        started = asyncio.Event()

        async def f(ev: asyncio.Event) -> None:
            started.set()
            await ev.wait()

        ev1 = asyncio.Event()
        # executing
        fut1 = await executor.submit(f(ev1))

        ev2 = asyncio.Event()
        # pending
        fut2 = await executor.submit(f(ev2))

        await started.wait()
        asyncio.get_running_loop().call_later(0.01, ev1.set)

        await executor.shutdown(cancel_futures=True)

        self.assertTrue(fut1.done())
        self.assertIsNone(fut1.result())

        self.assertTrue(fut2.cancelled())

    async def test_shutdown_no_wait(self) -> None:
        executor = self.make_executor(1)
        started = asyncio.Event()

        async def f() -> None:
            started.set()
            await asyncio.sleep(60)

        fut = await executor.submit(f())
        # wait to put submitted request into a worker
        await started.wait()

        await executor.shutdown(wait=False)

        self.assertTrue(fut.cancelled())

    async def test_shutdown_wt_exception_from_worker(self) -> None:
        executor = self.make_executor()
        executor._lazy_init()

        # emulate unhandled error by putting bad data into the queue
        await executor._work_items.put(None)  # type: ignore[arg-type]

        ok = False
        try:
            await executor.shutdown()
        except* AttributeError:
            ok = True

        self.assertTrue(ok)


class TestCancellation(BaseTestCase):
    async def test_cancelling_future_cancels_task(self) -> None:
        executor = self.make_executor()
        cancelled = asyncio.Event()
        started = asyncio.Event()

        async def f() -> None:
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        fut = await executor.submit(f())
        await started.wait()

        fut.cancel()
        await cancelled.wait()

    async def test_cancelling_map_cancels_tasks(self) -> None:
        executor = self.make_executor()
        cancelled = set()
        ev = asyncio.Event()

        async def f(i: int) -> None:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.add(i)
                if len(cancelled) == 5:
                    ev.set()
                raise

        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.01):
                async for _ in executor.map(f, range(5)):
                    pass

        await ev.wait()
        self.assertEqual(cancelled, {0, 1, 2, 3, 4})


class TestExceptions(BaseTestCase):
    async def test_dont_execute_with_done_future(self) -> None:
        executor = self.make_executor(1)
        started = asyncio.Event()

        async def f(num: int, ev: asyncio.Event) -> int:
            started.set()
            await ev.wait()
            return num

        ev1 = asyncio.Event()
        # executing
        fut1 = await executor.submit(f(1, ev1))

        ev2 = asyncio.Event()
        # pending
        fut2 = await executor.submit(f(2, ev2))

        # wait to put submitted request into a worker
        await started.wait()

        # Setting the result is strange, user should never do it.
        # But the executor should not crash at least
        fut2.set_result(10)

        ev1.set()
        ev2.set()

        self.assertEqual(await fut1, 1)
        self.assertEqual(await fut2, 10)

    async def test_dont_override_done_future(self) -> None:
        executor = self.make_executor()
        started = asyncio.Event()

        async def f(num: int, ev: asyncio.Event) -> int:
            started.set()
            await ev.wait()
            return num

        ev = asyncio.Event()
        fut = await executor.submit(f(1, ev))

        # wait to put submitted request into a worker
        await started.wait()

        # Setting the result is strange, user should never do it.
        # But the executor should not crash at least
        fut.set_result(10)
        ev.set()

        self.assertEqual(await fut, 10)

    async def test_coro_raises_exception(self) -> None:
        executor = self.make_executor()

        async def f() -> None:
            raise Exception("test exception")

        fut = await executor.submit(f())

        with self.assertRaisesRegex(Exception, "test exception"):
            await fut

    async def test_dont_override_exception_in_future(self) -> None:
        executor = self.make_executor()
        started = asyncio.Event()

        async def f(ev: asyncio.Event) -> int:
            started.set()
            await ev.wait()
            raise Exception("test exception")

        ev = asyncio.Event()
        fut = await executor.submit(f(ev))

        # wait to put submitted request into a worker
        await started.wait()

        # Setting the result is strange, user should never do it.
        # But the executor should not crash at least
        fut.set_exception(Exception("override"))
        ev.set()

        with self.assertRaisesRegex(Exception, "override"):
            await fut


class TestTaskNames(BaseTestCase):
    async def test_worker_name(self) -> None:
        executor = self.make_executor()
        executor._lazy_init()
        self.assertRegex(
            executor._tasks[0].get_name(), r"Executor-(\d+)_(\d+)"
        )

    async def test_submit_name(self) -> None:
        executor = self.make_executor()

        async def f() -> str:
            task = asyncio.current_task()
            assert task is not None
            return task.get_name()

        ret = await (await executor.submit(f()))
        self.assertRegex(ret, rf"Executor-(\d+)_(\d+)\[{f.__qualname__}\]")

    async def test_custom_name(self) -> None:
        executor = self.make_executor(task_name_prefix="custom")

        async def f() -> str:
            task = asyncio.current_task()
            assert task is not None
            return task.get_name()

        ret = await (await executor.submit(f()))
        self.assertRegex(ret, rf"custom_(\d+)\[{f.__qualname__}\]")


class TestRateLimiter(BaseTestCase):
    async def test_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "max_throughput must be non-negative number"
        ):
            _RateLimiter(asyncio.get_event_loop(), -1)
        with self.assertRaisesRegex(
            ValueError, "time_window must be positive number"
        ):
            _RateLimiter(asyncio.get_event_loop(), 1, 0)


if __name__ == "__main__":
    unittest.main()
