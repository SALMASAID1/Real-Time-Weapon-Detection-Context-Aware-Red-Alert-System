/**
 * ui/src/components/alerts/ThreatLogTable.jsx
 *
 * Scrollable, filterable table of ThreatEvent records.
 * Used in both the side panel (LiveMonitor, limited rows) and
 * the full ThreatHistory page (paginated).
 *
 * Props:
 *   events   : ThreatEvent[]  — events to display
 *   compact  : boolean        — true → side panel mode (fewer columns)
 *   onRowClick : function     — optional callback when a row is clicked
 */
import React from 'react';
import ThreatBadge from '../monitor/ThreatBadge';

function formatTs(ts) {
  // Handle both Unix timestamp (number) and ISO string (from REST API)
  if (typeof ts === 'string') {
    return new Date(ts).toLocaleTimeString('en-GB', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  }
  return new Date(ts * 1000).toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

// Safely extract class_name from either WS events (nested) or REST events (top-level)
function getClassName(ev) {
  return ev.detection?.class_name ?? ev.class_name ?? 'Unknown';
}

// Safely extract confidence
function getConfidence(ev) {
  return ev.detection?.confidence ?? ev.confidence ?? 0;
}

export default function ThreatLogTable({ events = [], compact = false, onRowClick }) {
  if (events.length === 0) {
    return (
      <div className="empty-state">
        <span>No threat events logged yet.</span>
      </div>
    );
  }

  return (
    <div className="threat-table-wrap">
      <table className="threat-table" aria-label="Threat event log">
        <thead>
          <tr>
            <th>Time</th>
            {!compact && <th>Camera</th>}
            <th>Class</th>
            <th>Level</th>
            <th>Score</th>
            {!compact && <th>IoU</th>}
            {!compact && <th>Preview</th>}
          </tr>
        </thead>
        <tbody>
          {events.map((ev) => (
            <tr
              key={ev.event_id}
              className={ev.threat_level === 'HIGH' ? 'row--high' : ''}
              onClick={() => onRowClick?.(ev)}
              style={{ cursor: onRowClick ? 'pointer' : 'default' }}
            >
              <td className="col-mono">{formatTs(ev.timestamp)}</td>
              {!compact && <td className="col-mono">{ev.camera_id}</td>}
              <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                {getClassName(ev)}
              </td>
              <td>
                <ThreatBadge level={ev.threat_level} score={ev.composite_score} />
              </td>
              <td className="col-mono">
                {Math.round((ev.composite_score ?? 0) * 100)}%
              </td>
              {!compact && (
                <td className="col-mono">
                  {Math.round((ev.proximity_iou ?? 0) * 100)}%
                </td>
              )}
              {!compact && (
                <td className="col-mono" style={{ color: 'var(--text-muted)' }}>
                  {/* Snapshot preview — requires snapshot storage (planned) */}
                  —
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
