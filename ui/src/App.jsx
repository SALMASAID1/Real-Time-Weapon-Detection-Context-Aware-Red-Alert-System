/**
 * ui/src/App.jsx
 *
 * Root router shell. React Router keeps the WS connection in LiveMonitor
 * alive when navigating to History or Settings — critical for not dropping
 * the inference stream mid-session.
 *
 * Route map:
 *   /          → LiveMonitor  (live canvas + side panel)
 *   /history   → ThreatHistory (paginated REST table)
 *   /settings  → Settings (threshold config)
 */
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar       from './components/layout/Sidebar';
import LiveMonitor   from './pages/LiveMonitor';
import ThreatHistory from './pages/ThreatHistory';
import Settings      from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <Routes>
          <Route path="/"         element={<LiveMonitor   />} />
          <Route path="/history"  element={<ThreatHistory />} />
          <Route path="/settings" element={<Settings      />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
