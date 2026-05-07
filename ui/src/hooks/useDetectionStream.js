/**
 * ui/src/hooks/useDetectionStream.js
 *
 * Parses raw WebSocketFrame messages from useWebSocket into structured
 * state consumed by UI components.
 *
 * Separates concerns:
 *   useWebSocket    → raw WS lifecycle (connect/disconnect/message)
 *   useDetectionStream → domain parsing (frame, detections, threat state)
 *
 * State returned:
 *   frame         : { id, timestamp, fps, frameB64 }
 *   detections    : Detection[]   — all detections this frame
 *   threatEvents  : ThreatEvent[] — scored weapon events (weapon classes only)
 *   threatLevel   : 'NONE'|'LOW'|'HIGH'
 *   gradcamB64    : string|null   — base64 Grad-CAM JPEG or null
 *
 * Usage:
 *   const { frame, detections, threatLevel, gradcamB64 } =
 *     useDetectionStream(lastMessage);
 *
 * @param {object|null} rawMessage — The parsed JSON object from useWebSocket.lastMessage
 */
import { useState, useEffect } from 'react';

const DEFAULT_STATE = {
  frame:       null,
  detections:  [],
  threatEvents:[],
  threatLevel: 'NONE',
  gradcamB64:  null,
};

export function useDetectionStream(rawMessage) {
  const [state, setState] = useState(DEFAULT_STATE);

  useEffect(() => {
    if (!rawMessage) return;

    setState({
      frame: {
        id:        rawMessage.frame_id,
        timestamp: rawMessage.timestamp,
        fps:       rawMessage.fps,
        frameB64:  rawMessage.frame_b64,
      },
      detections:   rawMessage.detections    ?? [],
      threatEvents: rawMessage.threat_events ?? [],
      threatLevel:  rawMessage.threat_level  ?? 'NONE',
      gradcamB64:   rawMessage.gradcam_b64   ?? null,
    });
  }, [rawMessage]);

  return state;
}
