/**
 * ui/src/components/alerts/RedAlertBanner.jsx
 *
 * Full-viewport flash + floating banner on HIGH threat events.
 * Triggered by useAlertState.isAlertActive.
 *
 * Multi-modal alert:
 *   Visual: viewport red flash + sliding banner (CSS animations)
 *   Audio:  Web Audio API programmatic beep (no <audio> file dependency)
 *
 * The audio is generated via OscillatorNode — a short 880Hz sine tone.
 * Using the Web Audio API avoids requiring a WAV file asset and ensures
 * the beep fires immediately without buffering delay.
 *
 * Dismiss flow:
 *   1. Operator clicks "Acknowledge"
 *   2. dismissAlert() called → useAlertState enters cooldown
 *   3. PATCH /api/threats/{eventId}/ack fired (fire-and-forget)
 *
 * Props:
 *   isActive    : boolean
 *   event       : ThreatEvent | null
 *   onDismiss   : function
 */
import React, { useEffect } from 'react';
import { Siren } from 'lucide-react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function playAlertBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.6);
  } catch {
    // AudioContext blocked (no user gesture yet) — silent fail
  }
}

export default function RedAlertBanner({ isActive, event, onDismiss }) {
  // Play audio cue when banner first appears
  useEffect(() => {
    if (isActive) playAlertBeep();
  }, [isActive]);

  if (!isActive || !event) return null;

  const handleDismiss = () => {
    onDismiss();
    // Non-blocking: acknowledge on backend
    axios.patch(`${API_BASE}/api/threats/${event.event_id}/ack`).catch(() => {});
  };

  const className = event.detection?.class_name ?? event.class_name ?? 'Weapon';
  const conf  = Math.round((event.detection?.confidence ?? event.confidence ?? 0) * 100);
  const score = Math.round((event.composite_score ?? 0) * 100);

  return (
    <>
      {/* Full-viewport flash */}
      <div className="red-alert-overlay" aria-hidden="true" />

      {/* Floating banner */}
      <div
        className="red-alert-banner"
        role="alertdialog"
        aria-live="assertive"
        aria-label="Red Alert — High Priority Threat Detected"
      >
        <div className="red-alert-banner__icon">
          <Siren size={32} />
        </div>

        <div className="red-alert-banner__content">
          <div className="red-alert-banner__title">⚠ Red Alert — High Priority</div>
          <div className="red-alert-banner__detail">
            <strong>{className}</strong> detected on{' '}
            <strong>{event.camera_id}</strong><br />
            Confidence: {conf}% · Threat Score: {score}% · IoU: {((event.proximity_iou ?? 0) * 100).toFixed(0)}%
          </div>
        </div>

        <button
          className="red-alert-banner__dismiss"
          onClick={handleDismiss}
          aria-label="Acknowledge and dismiss alert"
        >
          Acknowledge
        </button>
      </div>
    </>
  );
}
