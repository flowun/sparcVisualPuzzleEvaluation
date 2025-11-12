import asyncio
import aiohttp
from typing import Callable, Optional, Any


class RequestQueueAsync:
    def __init__(
        self,
        max_concurrent_requests: int = 100,
        api_url: str = f"http://localhost:8000/v1/chat/completions",
        loop: Optional[asyncio.AbstractEventLoop] = None,
        client_timeout: Optional[float] = 10 * 60 * 60.0, # 10 hours
    ):
        self.api_url = api_url
        self.max_concurrent_requests = max_concurrent_requests
        self.loop = loop or asyncio.get_event_loop()
        self.queue: asyncio.Queue[tuple[dict, Callable[[Any, Any], Any]]] = asyncio.Queue(maxsize=self.max_concurrent_requests * 2)
        self.sem = asyncio.Semaphore(max_concurrent_requests)
        self._headers = {"Content-Type": "application/json"}
        self._worker_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._closed = False
        self._timeout = aiohttp.ClientTimeout(total=client_timeout) if client_timeout else None

    def monitor_queue(self):
        try:
            qsize = self.queue.qsize()
            qmax = self.queue.maxsize
        except Exception:
            qsize = -1
            qmax = -1
        sem_val = getattr(self.sem, "_value", None)
        if sem_val is not None:
            in_flight = max(0, self.max_concurrent_requests - sem_val)
        else:
            in_flight = None
        fullness = f"{qsize}/{qmax}" if qmax > 0 else f"{qsize}/unbounded"
        if in_flight is None:
            print(f"Queue: {fullness}, in_flight: unknown, waiting: {qsize}")
        else:
            print(f"Queue: {fullness}, in_flight: {in_flight}, waiting: {qsize}")

    async def start(self) -> None:
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                limit_per_host=self.max_concurrent_requests,
            )
            self._session = aiohttp.ClientSession(timeout=self._timeout, connector=connector)
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())
            self.loop = asyncio.get_running_loop()

    def add_request(self, payload: dict, callback_function: Callable[[Any, Any], Any]) -> None:
        asyncio.run_coroutine_threadsafe(
            self.queue.put((payload, callback_function)), self.loop
        )

    async def add_request_async(self, payload: dict, callback_function: Callable[[Any, Any], Any]) -> None:
        await self.queue.put((payload, callback_function))
        # self.monitor_queue()

    async def _process_queue(self) -> None:
        while not self._closed:
            payload, callback_function = await self.queue.get()
            # print(payload)
            asyncio.create_task(self._send_request(payload, callback_function))

    async def _send_request(self, payload: dict, callback_function: Callable[[Any, Any], Any]) -> None:
        assert self._session is not None, "Call start() before sending requests."
        async with self.sem:
            try:
                # print(type(payload))
                async with self._session.post(self.api_url, json=payload, headers={"Content-Type": "application/json"}) as response:
                    if response.status == 200:
                        output_payload = await response.json()
                        await self._invoke_callback(callback_function, output_payload, payload)
                    else:
                        await self._invoke_callback(callback_function, {"error": f"Status code {response.status}"}, payload)
            except Exception as e:
                print(e)
                await self._invoke_callback(callback_function, {"error": str(e)}, payload)
            finally:
                self.queue.task_done()

    async def _invoke_callback(self, callback_function: Callable[[Any, Any], Any], result: Any, input_payload: Any) -> None:
        if asyncio.iscoroutinefunction(callback_function):
            await callback_function(result, input_payload)
        else:
            await asyncio.to_thread(callback_function, result, input_payload)

    async def close(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()