/**
 * ui/src/components/layout/Sidebar.jsx
 * Extracted from App.jsx monolith. Owns navigation state via React Router.
 */
import React from 'react';
import { NavLink } from 'react-router-dom';
import { ShieldAlert, Activity, History, Settings } from 'lucide-react';

const NAV_ITEMS = [
  { to: '/',         icon: Activity,    label: 'Live Monitor' },
  { to: '/history',  icon: History,     label: 'Threat Log'   },
  { to: '/settings', icon: Settings,    label: 'Settings'     },
];

export default function Sidebar() {
  return (
    <nav className="sidebar" aria-label="Main navigation">
      <div className="sidebar__logo">
        <div className="sidebar__logo-icon">
          <ShieldAlert size={20} color="#ef4444" />
        </div>
        <div>
          <div className="sidebar__logo-text">Red Alert</div>
          <div className="sidebar__logo-sub">AWD &amp; TA System</div>
        </div>
      </div>

      <div className="sidebar__nav" role="list">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `nav-item${isActive ? ' nav-item--active' : ''}`
            }
            role="listitem"
          >
            <Icon size={18} aria-hidden="true" />
            {label}
          </NavLink>
        ))}
      </div>

      <div className="sidebar__footer">
        v1.0.0 · YOLOv11 + Swin<br />
        Academic Year 2025–2026
      </div>
    </nav>
  );
}
