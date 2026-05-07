/**
 * ui/src/components/settings/ThresholdPanel.jsx
 *
 * Live threshold configuration panel. Reads current settings from
 * GET /api/settings on mount and writes updates via PUT /api/settings.
 *
 * Props: none — self-contained, fetches and saves independently.
 */
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SLIDERS = [
  { key: 'iou_threshold',   label: 'Hand-Weapon IoU Threshold',
    desc: 'Minimum bounding-box overlap between a hand and weapon to count as "held".',
    min: 0.1, max: 0.9, step: 0.05 },
  { key: 'conf_threshold',  label: 'Detection Confidence Threshold',
    desc: 'Minimum model confidence to log a detection event.',
    min: 0.1, max: 0.9, step: 0.05 },
  { key: 'alert_threshold', label: 'Red Alert Score Threshold',
    desc: 'Composite score (Conf + IoU + Persistence) required to escalate to RED ALERT.',
    min: 0.3, max: 0.95, step: 0.05 },
];

const TOGGLES = [
  { key: 'telegram_enabled', label: 'Telegram Alert Notifications' },
  { key: 'audio_enabled',    label: 'Audio Alert Cue'              },
  { key: 'gradcam_on_high',  label: 'Generate Grad-CAM on HIGH Alerts' },
];

export default function ThresholdPanel() {
  const [settings, setSettings] = useState(null);
  const [saved,    setSaved]    = useState(false);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/settings`)
      .then(r => setSettings(r.data))
      .catch(() => setError('Could not reach backend.'));
  }, []);

  const handleSlider = (key, value) =>
    setSettings(s => ({ ...s, [key]: parseFloat(value) }));

  const handleToggle = (key) =>
    setSettings(s => ({ ...s, [key]: !s[key] }));

  const handleSave = () => {
    axios.put(`${API_BASE}/api/settings`, settings)
      .then(() => { setSaved(true); setTimeout(() => setSaved(false), 2500); })
      .catch(() => setError('Failed to save settings.'));
  };

  if (error)    return <p style={{ color: 'var(--accent-red)', padding: '2rem' }}>{error}</p>;
  if (!settings) return <p style={{ color: 'var(--text-muted)', padding: '2rem' }}>Loading settings…</p>;

  return (
    <div className="settings-grid">
      {/* Threshold sliders card */}
      <div className="settings-card">
        <div className="settings-card__title">Detection Thresholds</div>
        <div className="settings-card__desc">
          Adjust inference and alert sensitivity. Changes apply to the live
          inference engine immediately on Save — no restart required.
        </div>
        <div className="field-group">
          {SLIDERS.map(({ key, label, desc, min, max, step }) => (
            <div className="field" key={key}>
              <label className="field__label" htmlFor={`slider-${key}`}>
                <span>{label}</span>
                <span className="field__value">{settings[key].toFixed(2)}</span>
              </label>
              <input
                id={`slider-${key}`}
                className="field__slider"
                type="range"
                min={min} max={max} step={step}
                value={settings[key]}
                onChange={e => handleSlider(key, e.target.value)}
                aria-label={label}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Toggle switches card */}
      <div className="settings-card">
        <div className="settings-card__title">Alert Channels</div>
        <div className="settings-card__desc">
          Enable or disable individual alert delivery channels.
          Grad-CAM generation has a ~15ms latency cost per HIGH alert.
        </div>
        <div>
          {TOGGLES.map(({ key, label }) => (
            <div className="toggle-row" key={key}>
              <span className="toggle-row__label">{label}</span>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={settings[key]}
                  onChange={() => handleToggle(key)}
                  aria-label={label}
                />
                <span className="toggle__slider" />
              </label>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="save-btn" onClick={handleSave}>
            Save Settings
          </button>
          {saved && (
            <span style={{ color: 'var(--accent-green)', fontSize: '0.8rem', fontWeight: 600 }}>
              ✓ Applied live
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
