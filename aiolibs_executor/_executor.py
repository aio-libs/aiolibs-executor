from __future__ import annotations

import contextvars
import itertools
import sys
import threading
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Future,
    Task,
    gather,
    get_running_loop,
)
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
)
from types import TracebackType
from typing import Any, Generic, TypeVar, final, overload
from warnings import catch_warnings


# Use janus for now until aiologic is ready to implement QueueShutdown
# SEE: https://github.com/x42005e1f/aiologic/issues/7

from janus import Queue, QueueShutDown

if sys.version_info < (3, 10):
    from collections.abc import Awaitable, Sequence
    from typing import Protocol

    _T = TypeVar("_T")

    class _SupportsAnext(Protocol[_T]):
        async def __anext__(self) -> _T:
            pass

    def anext(it: _SupportsAnext[_T]) -> Awaitable:
        return it.__anext__()

    def aiter(it: Sequence[_T]) -> AsyncIterable[_T]:
        return it.__aiter__()


if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup
    from typing_extensions import Self
else:
    from typing import Self


R = TypeVar("R")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


@final
class Executor:
    _counter = itertools.count().__next__

    def __init__(
        self,
        num_workers: int = 0,
        *,
        max_pending: int = 0,
        task_name_prefix: str = "",
    ) -> None:
        if num_workers == 0:
            num_workers = 100
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than 0")
        if max_pending < 0:
            raise ValueError("max_pending must be non-negative number")
        self._num_workers = num_workers
        self._task_name_prefix = (
            task_name_prefix or f"Executor-{Executor._counter()}"
        )
        self._loop: AbstractEventLoop | None = None
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

    def submit_nowait(
        self,
        coro: Coroutine[Any, Any, R],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        loop = self._lazy_init()
        work_item = _WorkItem(coro, loop, context)
        self._work_items.async_q.put_nowait(work_item)
        return work_item.future

    async def submit(
        self,
        coro: Coroutine[Any, Any, R],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> Future[R]:
        loop = self._lazy_init()
        work_item = _WorkItem(coro, loop, context)
        await self._work_items.async_q.put(work_item)
        return work_item.future

    @overload
    def map(
        self,
        fn: Callable[[T1], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map(
        self,
        fn: Callable[[T1, T2], Coroutine[Any, Any, R]],
        it1: Iterable[T1],
        it2: Iterable[T2],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...

    @overload
    def map(
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
    def map(
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
    def map(
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

    async def map(
        self,
        fn: Callable[..., Coroutine[Any, Any, R]],
        iterable: Iterable[Any],
        /,
        *iterables: Iterable[Any],
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]:
        loop = self._lazy_init()
        work_items: list[_WorkItem[R]] = []
        for args in zip(iterable, *iterables, strict=False):
            work_item = _WorkItem(fn(*args), loop, context)
            await self._work_items.async_q.put(work_item)
            work_items.append(work_item)
        async for ret in self._process_items(work_items):
            yield ret

    @overload
    def amap(
        self,
        fn: Callable[[T1], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...
    @overload
    def amap(
        self,
        fn: Callable[[T1, T2], Coroutine[Any, Any, R]],
        it1: AsyncIterable[T1],
        it2: AsyncIterable[T2],
        /,
        *,
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]: ...
    @overload
    def amap(
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
    def amap(
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
    def amap(
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
    async def amap(
        self,
        fn: Callable[..., Coroutine[Any, Any, R]],
        iterable: AsyncIterable[Any],
        /,
        *iterables: AsyncIterable[Any],
        context: contextvars.Context | None = None,
    ) -> AsyncIterator[R]:
        loop = self._lazy_init()
        work_items: list[_WorkItem[R]] = []
        its = [aiter(iterable)] + [aiter(ait) for ait in iterables]
        while True:
            try:
                args = [await anext(it) for it in its]
                work_item = _WorkItem(fn(*args), loop, context)
                await self._work_items.async_q.put(work_item)
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
                self._work_items.async_q.get_nowait().cancel()

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

    def _lazy_init(self) -> AbstractEventLoop:
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        if self._loop is not None:
            try:
                loop = get_running_loop()
            except RuntimeError:
                # do nothing and reuse previously stored self._loop
                # to allow .submit_nowait() call from non-asyncio code
                return self._loop
            else:
                # the loop check technique is borrowed from asyncio.locks.
                if loop is not self._loop:
                    raise RuntimeError(
                        f"{self!r} is bound to a different event loop"
                    )
                return loop
        else:
            loop = get_running_loop()
            with _global_lock:
                # double-checked locking has a very low chance to have
                # self._loop assigned by another thread;
                # test suite doen't cover this case
                if self._loop is None:  # pragma: no branch
                    self._loop = loop
            for i in range(self._num_workers):
                task_name = self._task_name_prefix + f"_{i}"
                self._tasks.append(
                    loop.create_task(self._work(task_name), name=task_name)
                )
            return loop

    async def _process_items(
        self, work_items: list[_WorkItem[R]]
    ) -> AsyncIterator[R]:
        try:
            # NOTE: Polling future objects can be a bad apporch
            # callbacks need to be used in order to return items
            # in finishing order

            remaining = len(work_items)
            queue: Queue[Future[R]] = Queue()

            def on_done(fut: Future[R]) -> None:
                nonlocal queue, remaining
                queue.async_q.put_nowait(fut)
                remaining -= 1

            # No need to call for a copy,
            # loop will call it later
            for w in work_items:
                w.future.add_done_callback(on_done)

            while remaining or not queue.async_q.empty():
                fut = await queue.async_q.get()
                yield await fut

            # cleanup
            work_items.clear()

        except CancelledError:
            # The current task was cancelled, e.g. by timeout
            for work_item in work_items:
                work_item.cancel()
            raise

    async def _work(self, prefix: str) -> None:
        try:
            while True:
                worker = await self._work_items.async_q.get()
                await worker.execute(prefix)
        except QueueShutDown:
            pass


_global_lock = threading.Lock()


class _WorkItem(Generic[R]):
    __slots__ = (
        "coro",
        "loop",
        "context",
        "task",
        "future",
    )

    def __init__(
        self,
        coro: Coroutine[Any, Any, R],
        loop: AbstractEventLoop,
        context: contextvars.Context | None,
        task: Task[R] | None = None,
    ) -> None:
        self.coro = coro
        self.loop = loop
        self.context = context
        self.task = task
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
        if sys.version_info >= (3, 11):
            self.task = task = self.loop.create_task(  # type: ignore[call-arg]
                self.coro, context=self.context, name=name
            )
        # XXX: older versions of Python can't leverage context variables
        # Not handling it and letting the bad arguments run results in
        # a deadlock!
        else:
            self.task = task = self.loop.create_task(self.coro, name=name)

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
        self.future.cancel()
        self.cleanup()

    def cleanup(self) -> None:
        with catch_warnings(action="ignore", category=RuntimeWarning):  # type: ignore[call-overload]
            # Suppress RuntimeWarning: coroutine 'coro' was never awaited.
            # The warning is possible if .shutdown() was called
            # with cancel_futures=True and there are non-started coroutines
            # in pedning work_items list.
            del self.coro

    def done_callback(self, fut: Future[R]) -> None:
        if self.task is not None and fut.cancelled():
            self.task.cancel()
