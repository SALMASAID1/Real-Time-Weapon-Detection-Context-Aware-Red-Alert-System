/**
 * ui/src/hooks/useAlertState.js
 *
 * State machine for Red Alert escalation and dismissal.
 *
 * Alert lifecycle:
 *   NONE ──(threatLevel=HIGH)──▶ ACTIVE
 *   ACTIVE ──(operator dismisses)──▶ DISMISSED
 *   DISMISSED ──(2s cooldown, then new HIGH)──▶ ACTIVE
 *
 * Why a cooldown after dismiss:
 *   Without a cooldown, a persistent weapon detection (high Persistence score)
 *   would immediately re-trigger the banner after the operator dismisses it.
 *   A 2-second cooldown gives the operator time to acknowledge and take action.
 *
 * State returned:
 *   isAlertActive    : boolean       — Whether the Red Alert banner should render
 *   activeEvent      : ThreatEvent|null — The highest-score event driving the alert
 *   dismissAlert     : function      — Call from RedAlertBanner dismiss button
 *
 * @param {string}      threatLevel  — Current frame's threat level
 * @param {ThreatEvent[]} threatEvents — Scored events from useDetectionStream
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const DISMISS_COOLDOWN_MS = 2000;

export function useAlertState(threatLevel, threatEvents) {
  const [isAlertActive, setIsAlertActive] = useState(false);
  const [activeEvent,   setActiveEvent]   = useState(null);
  const cooldownRef = useRef(false);

  // Escalate when a HIGH event arrives and we're not in cooldown
  useEffect(() => {
    if (threatLevel !== 'HIGH' || cooldownRef.current) return;

    // Pick the highest composite-score event to surface in the banner
    const highEvents = threatEvents.filter(e => e.threat_level === 'HIGH');
    if (highEvents.length === 0) return;

    const top = highEvents.reduce((a, b) =>
      b.composite_score > a.composite_score ? b : a
    );

    setActiveEvent(top);
    setIsAlertActive(true);
  }, [threatLevel, threatEvents]);

  const dismissAlert = useCallback(() => {
    setIsAlertActive(false);
    setActiveEvent(null);
    cooldownRef.current = true;
    setTimeout(() => { cooldownRef.current = false; }, DISMISS_COOLDOWN_MS);
  }, []);

  return { isAlertActive, activeEvent, dismissAlert };
}
