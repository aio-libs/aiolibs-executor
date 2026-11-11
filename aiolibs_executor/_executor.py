import contextvars
import dataclasses
import itertools
import sys
import threading
from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Future,
    Queue,
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
from typing import Any, Generic, Self, TypeVar, final, overload
from warnings import catch_warnings


if sys.version_info < (3, 13):
    # Utilize a backported QueueShutdown mechanism
    class QueueShutDown(Exception):
        """Raised when putting on to or getting from a shut-down Queue."""

    from asyncio import sleep
else:
    from asyncio import QueueShutDown

R = TypeVar("R")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")


class _WorkerQueue(Queue):
    """Subclass of `asyncio.Queue` that provides backported functionality
    for older python versions."""

    if sys.version_info < (3, 13):
        __slots__ = (
            "_maxsize",
            "_getters",
            "_putters",
            "_unfinished_tasks",
            "_finished",
            "_is_shutdown",
        )

        def __init__(self, maxsize=0):
            super().__init__(maxsize)
            self._is_shutdown = False

        async def put(self, item):
            """Put an item into the queue.

            Put an item into the queue. If the queue is full, wait until a free
            slot is available before adding item.

            Raises QueueShutDown if the queue has been shut down.
            """
            force_checkpoint = not self.full()
            while self.full():
                if self._is_shutdown:
                    raise QueueShutDown
                putter = self._get_loop().create_future()
                self._putters.append(putter)
                try:
                    await putter
                except:
                    putter.cancel()  # Just in case putter is not done yet.
                    try:
                        # Clean self._putters from canceled putters.
                        self._putters.remove(putter)
                    except ValueError:
                        # The putter could be removed from self._putters by a
                        # previous get_nowait call or a shutdown call.
                        pass
                    if not self.full() and not putter.cancelled():
                        # We were woken up by get_nowait(), but can't take
                        # the call.  Wake up the next in line.
                        self._wakeup_next(self._putters)
                    raise
            self.put_nowait(item)
            # XXX: coroutines in 3.13 have slightly different behaviors
            # so to emulate task movement a checkpoint is forced in.
            if force_checkpoint:
                await sleep(0)

        def put_nowait(self, item):
            """Put an item into the queue without blocking.

            If no free slot is immediately available, raise QueueFull.

            Raises QueueShutDown if the queue has been shut down.
            """
            if self._is_shutdown:
                raise QueueShutDown
            super().put_nowait(item)

        async def get(self):
            """Remove and return an item from the queue.

            If queue is empty, wait until an item is available.

            Raises QueueShutDown if the queue has been shut down and is empty,
            or if the queue has been shut down immediately.
            """
            checkpoint_needed = not self.empty()
            while self.empty():
                if self._is_shutdown:
                    raise QueueShutDown
                getter = self._get_loop().create_future()
                self._getters.append(getter)
                try:
                    await getter
                except:
                    getter.cancel()  # Just in case getter is not done yet.
                    try:
                        # Clean self._getters from canceled getters.
                        self._getters.remove(getter)
                    except ValueError:
                        # The getter could be removed from self._getters by a
                        # previous put_nowait call, or a shutdown call.
                        pass
                    if not self.empty() and not getter.cancelled():
                        # We were woken up by put_nowait(), but can't take
                        # the call.  Wake up the next in line.
                        self._wakeup_next(self._getters)
                    raise

            item = self.get_nowait()
            if checkpoint_needed:
                await sleep(0)
            return item

        def get_nowait(self):
            """Remove and return an item from the queue.

            Return an item if one is immediately available, else raise
            QueueEmpty.

            Raises QueueShutDown if the queue has been shut down and is empty,
            or if the queue has been shut down immediately.
            """
            if self._is_shutdown:
                raise QueueShutDown
            return super().get_nowait()

        def shutdown(self, immediate=False):
            """Shut-down the queue, making queue gets and puts raise
            QueueShutDown.

            By default, gets will only raise once the queue is empty. Set
            'immediate' to True to make gets raise immediately instead.

            All blocked callers of put() and get() will be unblocked.

            If 'immediate', the queue is drained and unfinished tasks
            is reduced by the number of drained tasks.  If unfinished tasks
            is reduced to zero, callers of Queue.join are unblocked.
            """
            self._is_shutdown = True
            if immediate:
                while not self.empty():
                    self._get()
                    if self._unfinished_tasks > 0:
                        self._unfinished_tasks -= 1
                if self._unfinished_tasks == 0:
                    self._finished.set()
            # All getters need to re-check queue-empty to raise ShutDown
            while self._getters:
                getter = self._getters.popleft()
                if not getter.done():
                    getter.set_result(None)
            while self._putters:
                putter = self._putters.popleft()
                if not putter.done():
                    putter.set_result(None)

        def __repr__(self):
            result = super().__repr__()
            if self._is_shutdown:
                result += " shutdown"
            return result


@final
class Executor:
    _counter = itertools.count().__next__
    __slots__ = (
        "__weakref__",
        "_num_workers",
        "_task_name_prefix",
        "_loop",
        "_shutdown",
        "_work_items",
        "_tasks",
    )

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
        self._work_items: _WorkerQueue[_WorkItem[Any]] = _WorkerQueue(
            max_pending
        )
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
        self._work_items.put_nowait(work_item)
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
        await self._work_items.put(work_item)
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
            await self._work_items.put(work_item)
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


@dataclasses.dataclass(slots=True)
class _WorkItem(Generic[R]):
    coro: Coroutine[Any, Any, R]
    loop: AbstractEventLoop
    context: contextvars.Context | None
    task: Task[R] | None = None
    future: Future[R] = dataclasses.field(init=False)

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
