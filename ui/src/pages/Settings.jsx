/**
 * ui/src/pages/Settings.jsx
 * Settings page — wraps ThresholdPanel with header.
 */
import React from 'react';
import Header          from '../components/layout/Header';
import ThresholdPanel  from '../components/settings/ThresholdPanel';

export default function Settings() {
  return (
    <div className="main-content">
      <Header title="System Settings" isConnected={false} />
      <div className="page">
        <div>
          <h2 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            Detection &amp; Alert Configuration
          </h2>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            All threshold changes apply to the live inference engine immediately. No restart required.
          </p>
        </div>
        <ThresholdPanel />
      </div>
    </div>
  );
}
