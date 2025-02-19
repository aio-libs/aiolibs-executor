import contextvars
import dataclasses
import itertools
import threading
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Condition,
    Future,
    Queue,
    QueueShutDown,
    Task,
    gather,
    get_running_loop,
    sleep,
)
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
)
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Self, final, overload
from warnings import catch_warnings


@final
class Executor:
    _counter = itertools.count().__next__

    def __init__(
        self,
        num_workers: int = 0,
        *,
        max_pending: int = 0,
        max_throughput: int = 0,
        task_name_prefix: str = "",
    ) -> None:
        if num_workers == 0:
            num_workers = 100
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than 0")
        if max_pending < 0:
            raise ValueError("max_pending must be non-negative number")
        if max_throughput < 0:
            raise ValueError("max_throughput must be non-negative number")
        self._num_workers = num_workers
        self._max_throughput = max_throughput
        self._task_name_prefix = (
            task_name_prefix or f"Executor-{Executor._counter()}"
        )
        self._loop: AbstractEventLoop | None = None
        self._rate_limiter: _RateLimiter | None = None
        self._shutdown = False
        self._work_items: Queue[_WorkItem[Any]] = Queue(max_pending)
        # tasks are much cheaper than threads or processes,
        # there is no need for adjusting tasks count on the fly like
        # ThreadPoolExecutor or ProcessPoolExecutor do.
        self._tasks: list[Task[None]] = []

    async def __aenter__(self) -> Self:
        self._lazy_init()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    def submit_nowait[R](
        self,
        coro: Coroutine[Any, Any, R],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        loop, rate_limiter = self._lazy_init()
        work_item = _WorkItem(coro, loop, rate_limiter, context)
        self._work_items.put_nowait(work_item)
        return work_item.future

    async def submit[R](
        self,
        coro: Coroutine[Any, Any, R],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        loop, rate_limiter = self._lazy_init()
        work_item = _WorkItem(coro, loop, rate_limiter, context)
        await self._work_items.put(work_item)
        return work_item.future

    @overload
    def map[R, T1](
        self,
        fn: Callable[[T1], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map[R, T1, T2](
        self,
        fn: Callable[[T1, T2], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        it2: Iterable[T2],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map[R, T1, T2, T3](
        self,
        fn: Callable[[T1, T2, T3], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        it2: Iterable[T2],
        it3: Iterable[T3],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map[R, T1, T2, T3, T4](
        self,
        fn: Callable[[T1, T2, T3, T4], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        it2: Iterable[T2],
        it3: Iterable[T3],
        it4: Iterable[T4],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map[R, T1, T2, T3, T4, T5](
        self,
        fn: Callable[[T1, T2, T3, T4, T5], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        it2: Iterable[T2],
        it3: Iterable[T3],
        it4: Iterable[T4],
        it5: Iterable[T5],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    async def map[R](
        self,
        fn: Callable[..., Coroutine[Any, Any, R]],
        iterable: Iterable[Any],
        /,
        *iterables: Iterable[Any],
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]:
        loop, rate_limiter = self._lazy_init()
        work_items: list[_WorkItem[R]] = []
        for args in zip(iterable, *iterables, strict=False):
            work_item = _WorkItem(fn(*args), loop, rate_limiter, context)
            await self._work_items.put(work_item)
            work_items.append(work_item)
        async for ret in self._process_items(work_items):
            yield ret

    @overload
    def amap[R, T1](
        self,
        fn: Callable[[T1], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def amap[R, T1, T2](
        self,
        fn: Callable[[T1, T2], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        it2: AsyncIterable[T2],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def amap[R, T1, T2, T3](
        self,
        fn: Callable[[T1, T2, T3], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        it2: AsyncIterable[T2],
        it3: AsyncIterable[T3],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def amap[R, T1, T2, T3, T4](
        self,
        fn: Callable[[T1, T2, T3, T4], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        it2: AsyncIterable[T2],
        it3: AsyncIterable[T3],
        it4: AsyncIterable[T4],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def amap[R, T1, T2, T3, T4, T5](
        self,
        fn: Callable[[T1, T2, T3, T4, T5], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        it2: AsyncIterable[T2],
        it3: AsyncIterable[T3],
        it4: AsyncIterable[T4],
        it5: AsyncIterable[T5],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    async def amap[R](
        self,
        fn: Callable[..., Coroutine[Any, Any, R]],
        iterable: AsyncIterable[Any],
        /,
        *iterables: AsyncIterable[Any],
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]:
        loop, rate_limiter = self._lazy_init()
        work_items: list[_WorkItem[R]] = []
        its = [aiter(iterable)] + [aiter(ait) for ait in iterables]
        while True:
            try:
                args = [await anext(it) for it in its]
                work_item = _WorkItem(fn(*args), loop, rate_limiter, context)
                await self._work_items.put(work_item)
                work_items.append(work_item)
            except StopAsyncIteration:
                break
        async for ret in self._process_items(work_items):
            yield ret

    async def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._loop is None:
            return
        if cancel_futures:
            # Drain all work items from the queue, and then cancel their
            # associated futures.
            while not self._work_items.empty():
                self._work_items.get_nowait().cancel()

        self._work_items.shutdown()
        if not wait:
            for task in self._tasks:
                task.cancel()

        rets = await gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()  # cleanup cycle references
        excs = [
            exc
            for exc in rets
            if isinstance(exc, BaseException)
            and type(exc) is not CancelledError
        ]
        if excs:
            try:
                raise BaseExceptionGroup(
                    "unhandled errors during Executor.shutdown()",
                    excs,
                ) from None
            finally:
                del excs

    def _lazy_init(self) -> tuple[AbstractEventLoop, "_RateLimiter"]:
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        if self._loop is not None:
            if self._rate_limiter is None:
                raise RuntimeError(f"{self!r} failed to fully initialize")
            try:
                loop = get_running_loop()
            except RuntimeError:
                # do nothing and reuse previously stored self._loop
                # to allow .submit_nowait() call from non-asyncio code
                return self._loop, self._rate_limiter
            else:
                # the loop check technique is borrowed from asyncio.locks.
                if loop is not self._loop:
                    raise RuntimeError(
                        f"{self!r} is bound to a different event loop"
                    )
                return loop, self._rate_limiter
        else:
            loop = get_running_loop()
            with _global_lock:
                # double-checked locking has a very low chance to have
                # self._loop assigned by another thread;
                # test suite doesn't cover this case
                if self._loop is None:  # pragma: no branch
                    self._loop = loop
                self._rate_limiter = _RateLimiter(
                    loop,
                    self._max_throughput,
                )
            for i in range(self._num_workers):
                task_name = self._task_name_prefix + f"_{i}"
                self._tasks.append(
                    loop.create_task(self._work(task_name), name=task_name)
                )
            return loop, self._rate_limiter

    async def _process_items[R](
        self, work_items: list["_WorkItem[R]"]
    ) -> AsyncIterator[R]:
        try:
            # reverse to keep finishing order
            work_items.reverse()
            while work_items:
                # Careful not to keep a reference to the popped future
                yield await work_items.pop().future
        except CancelledError:
            # The current task was cancelled, e.g. by timeout
            for work_item in work_items:
                work_item.cancel()
            raise

    async def _work(self, prefix: str) -> None:
        try:
            while True:
                await (await self._work_items.get()).execute(prefix)
        except QueueShutDown:
            pass


_global_lock = threading.Lock()


class _RateLimiter(AbstractAsyncContextManager["_RateLimiter"]):
    def __init__(
        self,
        event_loop: AbstractEventLoop,
        max_throughput: int = 0,
        time_window: float = 1.0,
    ) -> None:
        if max_throughput < 0:
            raise ValueError("max_throughput must be non-negative number")
        if time_window <= 0:
            raise ValueError("time_window must be positive number")
        super().__init__()
        self._event_loop = event_loop
        self._throttling_lock = Condition()
        self._max_throughput = max_throughput
        self._time_window = time_window
        self._tracked_tasks = 0

    async def __aenter__(self) -> Self:
        async with self._throttling_lock:
            await self._throttling_lock.wait_for(self._allow_entry)
            self._tracked_tasks += 1

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._event_loop.create_task(self._remove_tracked_task())

    def _allow_entry(self) -> bool:
        return (
            self._max_throughput == 0
            or self._tracked_tasks < self._max_throughput
        )

    async def _remove_tracked_task(self) -> None:
        await sleep(self._time_window)
        async with self._throttling_lock:
            self._tracked_tasks -= 1
            self._throttling_lock.notify_all()


@dataclasses.dataclass
class _WorkItem[R]:
    coro: Coroutine[Any, Any, R]
    loop: AbstractEventLoop
    rate_limiter: _RateLimiter
    context: contextvars.Context | None
    task: Task[R] | None = None

    def __post_init__(self) -> None:
        self.future: Future[R] = self.loop.create_future()

    async def execute(self, prefix: str) -> None:
        fut = self.future
        if fut.done():
            self.cleanup()
            return
        name = prefix
        try:
            name += f"[{self.coro.__qualname__}]"
        except AttributeError:  # pragma: no cover
            # Some custom coroutines and mocks could not have __qualname__,
            # don't add a suffix in this case.
            pass
        async with self.rate_limiter:
            self.task = task = self.loop.create_task(
                self.coro, context=self.context, name=name
            )
            fut.add_done_callback(self.done_callback)
            try:
                ret = await task
            except CancelledError:
                fut.cancel()
            except BaseException as ex:
                if not fut.done():
                    fut.set_exception(ex)
            else:
                if not fut.done():
                    fut.set_result(ret)

    def cancel(self) -> None:
        fut = self.future
        fut.cancel()
        self.cleanup()

    def cleanup(self) -> None:
        with catch_warnings(action="ignore", category=RuntimeWarning):
            # Suppress RuntimeWarning: coroutine 'coro' was never awaited.
            # The warning is possible if .shutdown() was called
            # with cancel_futures=True and there are non-started coroutines
            # in pedning work_items list.
            del self.coro

    def done_callback(self, fut: Future[R]) -> None:
        if self.task is not None and fut.cancelled():
            self.task.cancel()
