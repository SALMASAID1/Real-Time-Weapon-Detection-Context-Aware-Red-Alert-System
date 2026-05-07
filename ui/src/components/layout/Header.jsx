/**
 * ui/src/components/layout/Header.jsx
 * Top bar: page title, backend status indicator, notification icon.
 */
import React from 'react';
import { Bell } from 'lucide-react';

/**
 * @param {string}  title      — Page title (e.g. "Live Surveillance")
 * @param {boolean} isConnected — WS connection state from useWebSocket
 * @param {number}  alertCount  — Unacknowledged HIGH alert count for badge
 */
export default function Header({ title, isConnected, alertCount = 0 }) {
  return (
    <header className="header" role="banner">
      <div className="header__title">
        <span>{title}</span>
      </div>

      <div className="header__actions">
        {/* Notification bell with badge */}
        <div style={{ position: 'relative', color: 'var(--text-secondary)' }}>
          <Bell size={20} aria-label={`${alertCount} unacknowledged alerts`} />
          {alertCount > 0 && (
            <span style={{
              position: 'absolute', top: '-6px', right: '-6px',
              background: 'var(--accent-red)', color: '#fff',
              borderRadius: '9999px', fontSize: '0.6rem',
              fontWeight: 700, width: '16px', height: '16px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {alertCount > 9 ? '9+' : alertCount}
            </span>
          )}
        </div>

        {/* Backend connectivity status */}
        <div
          className={`status-badge status-badge--${isConnected ? 'active' : 'inactive'}`}
          role="status"
          aria-live="polite"
        >
          <span className="status-badge__dot" />
          {isConnected ? 'Stream Active' : 'Connecting…'}
        </div>
      </div>
    </header>
  );
}
