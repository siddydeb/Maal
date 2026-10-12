"""_wait_ws tolerates Comfy going unresponsive while it loads a big model."""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from backend import comfy_client
from backend.comfy_client import ComfyClient, ComfyUIError
from backend.config import Settings


def _client() -> ComfyClient:
    return ComfyClient(Settings(comfyui_url="http://comfy.test:8188"))


class WaitGraceTests(unittest.TestCase):
    def test_open_timeouts_within_grace_keep_waiting(self):
        client = _client()
        finished = iter([False, False, False, True])

        async def prompt_finished(_pid):
            return next(finished)

        def connect(*_a, **_kw):
            raise TimeoutError("timed out during opening handshake")

        with mock.patch.object(client, "_prompt_finished", side_effect=prompt_finished), \
                mock.patch.object(comfy_client.websockets, "connect", side_effect=connect), \
                mock.patch.object(comfy_client.asyncio, "sleep", new=mock.AsyncMock()):
            asyncio.run(client._wait_ws("cid", "pid", timeout=600))

    def test_gives_up_after_grace(self):
        client = _client()
        clock = iter(range(0, 10_000, 100))
        loop = asyncio.new_event_loop()

        async def prompt_finished(_pid):
            return False

        def connect(*_a, **_kw):
            raise TimeoutError("timed out during opening handshake")

        with mock.patch.object(client, "_prompt_finished", side_effect=prompt_finished), \
                mock.patch.object(comfy_client.websockets, "connect", side_effect=connect), \
                mock.patch.object(comfy_client.asyncio, "sleep", new=mock.AsyncMock()), \
                mock.patch.object(loop, "time", side_effect=lambda: float(next(clock))):
            with self.assertRaises(ComfyUIError):
                loop.run_until_complete(client._wait_ws("cid", "pid", timeout=100_000))
        loop.close()


if __name__ == "__main__":
    unittest.main()
