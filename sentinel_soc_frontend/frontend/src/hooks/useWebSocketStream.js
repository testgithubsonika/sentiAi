/**
 * hooks/useWebSocketStream.js
 * ============================
 * Generic, reusable WebSocket client hook. Connects once, auto-reconnects
 * with capped exponential backoff on drop, parses each frame as JSON, and
 * dispatches by the backend's `{ "type": "event" | "alert", ... }` shape
 * via `onMessage`. Exposes `status` for the header's connection pulse.
 *
 * Kept intentionally dumb (no alert-queue/chart state in here) so it can
 * be reused anywhere a raw stream is needed -- see
 * `context/AnomalyStreamProvider.jsx` for the app's single shared
 * connection + derived dashboard state.
 */

import { useEffect, useRef, useState, useCallback } from 'react';

const MAX_BACKOFF_MS = 15_000;
const BASE_BACKOFF_MS = 750;

export function useWebSocketStream(url, { onMessage, enabled = true } = {}) {
  const [status, setStatus] = useState('connecting'); // connecting | open | closed | error
  const socketRef = useRef(null);
  const attemptRef = useRef(0);
  const timeoutRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!enabled || !url) return;

    setStatus((prev) => (prev === 'open' ? prev : 'connecting'));
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => {
      attemptRef.current = 0;
      setStatus('open');
    };

    socket.onmessage = (event) => {
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return; // ignore malformed frames rather than crashing the stream
      }
      onMessageRef.current?.(payload);
    };

    socket.onerror = () => {
      setStatus('error');
    };

    socket.onclose = () => {
      setStatus('closed');
      if (!enabled) return;
      const delay = Math.min(BASE_BACKOFF_MS * 2 ** attemptRef.current, MAX_BACKOFF_MS);
      attemptRef.current += 1;
      timeoutRef.current = setTimeout(connect, delay);
    };
  }, [url, enabled]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(timeoutRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connect]);

  const send = useCallback((data) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  return { status, send };
}
