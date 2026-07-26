"""
websocket_manager.py
=====================
Tracks connected dashboard clients on `/ws/stream` and broadcasts JSON
messages (live events + alerts) to all of them concurrently. A dead/
disconnected socket found mid-broadcast is pruned rather than blowing
up the whole broadcast.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger("websocket_manager")


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("Client connected (%d total).", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("Client disconnected (%d total).", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Fan out `message` (JSON-serialized once) to every connected
        client. Sockets that fail to send are dropped."""
        if not self.active_connections:
            return
        payload = json.dumps(message, default=str)

        async with self._lock:
            targets = list(self.active_connections)

        results = await asyncio.gather(
            *(conn.send_text(payload) for conn in targets), return_exceptions=True
        )

        stale = [conn for conn, result in zip(targets, results) if isinstance(result, Exception)]
        if stale:
            async with self._lock:
                for conn in stale:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)
            logger.info("Pruned %d stale connections.", len(stale))

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()
