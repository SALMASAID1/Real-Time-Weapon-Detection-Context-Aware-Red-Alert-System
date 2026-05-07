/**
 * ui/src/components/monitor/ThreatBadge.jsx
 * Colour-coded pill showing current threat level.
 */
import React from 'react';
import { ShieldCheck, ShieldAlert, Siren } from 'lucide-react';

const CONFIG = {
  NONE: { icon: ShieldCheck, label: 'All Clear',   cls: 'threat-badge--none'  },
  LOW:  { icon: ShieldAlert, label: 'Low Threat',  cls: 'threat-badge--low'   },
  HIGH: { icon: Siren,       label: '⚠ RED ALERT', cls: 'threat-badge--high'  },
};

/**
 * @param {'NONE'|'LOW'|'HIGH'} level
 * @param {number} score — Composite threat score (0–1), shown as percentage
 */
export default function ThreatBadge({ level = 'NONE', score = 0 }) {
  const { icon: Icon, label, cls } = CONFIG[level] ?? CONFIG.NONE;
  return (
    <span className={`threat-badge ${cls}`} role="status" aria-live="polite">
      <Icon size={12} aria-hidden="true" />
      {label}
      {level !== 'NONE' && (
        <span style={{ opacity: 0.75 }}>
          &nbsp;{Math.round(score * 100)}%
        </span>
      )}
    </span>
  );
}
