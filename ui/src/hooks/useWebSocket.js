/**
 * ui/src/hooks/useWebSocket.js
 *
 * Manages the full WebSocket lifecycle for a single camera stream.
 *
 * Design rationale:
 *   - Single hook instance provides the connection to all consumers via Context.
 *     Do NOT call this hook in individual components — call it once at the
 *     App or LiveMonitor level and pass data down (or use WS Context).
 *   - Reconnect strategy: exponential backoff (1s → 2s → 4s → ... → 30s cap).
 *     Prevents hammering the FastAPI server when it restarts during development.
 *   - The WebSocket URL is built from the VITE_API_URL env var (set in .env):
 *       ws://localhost:8000/ws/stream/${cameraId}
 *
 * State returned:
 *   lastMessage : object | null  — Parsed JSON of the latest WebSocketFrame
 *   isConnected : boolean        — Whether the WS is in OPEN state
 *   reconnectCount : number      — How many times we have reconnected (for UI display)
 *
 * The hook does NOT expose a send() function — this is a receive-only stream.
 *
 * Usage (in LiveMonitor.jsx):
 *   const { lastMessage, isConnected } = useWebSocket('CAM-01');
 */

import { useState, useEffect, useRef, useCallback } from 'react';

const BASE_WS_URL = import.meta.env.VITE_API_WS_URL || 'ws://localhost:8000';
const MAX_BACKOFF_MS = 30_000;

export function useWebSocket(cameraId) {
  const [lastMessage,    setLastMessage]    = useState(null);
  const [isConnected,    setIsConnected]    = useState(false);
  const [reconnectCount, setReconnectCount] = useState(0);

  const wsRef      = useRef(null);
  const backoffRef = useRef(1000);
  const timerRef   = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const url = `${BASE_WS_URL}/ws/stream/${cameraId}`;
    const ws  = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setIsConnected(true);
      backoffRef.current = 1000; // reset backoff on successful connect
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      try {
        const frame = JSON.parse(event.data);
        setLastMessage(frame);
      } catch {
        // Malformed JSON — ignore silently
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setIsConnected(false);
      setReconnectCount(c => c + 1);
      // Schedule reconnect with exponential backoff
      timerRef.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
        connect();
      }, backoffRef.current);
    };

    ws.onerror = () => ws.close(); // onclose handles reconnect
  }, [cameraId]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { lastMessage, isConnected, reconnectCount };
}
