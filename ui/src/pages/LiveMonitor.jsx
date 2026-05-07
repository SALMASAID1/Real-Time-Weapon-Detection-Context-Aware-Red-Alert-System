/**
 * ui/src/pages/LiveMonitor.jsx
 *
 * Primary surveillance page.
 * Owns the WS connection — passes data down to VideoCanvas and the side panel.
 * Manages Red Alert state via useAlertState.
 */
import React, { useState } from 'react';
import { useWebSocket }        from '../hooks/useWebSocket';
import { useDetectionStream }  from '../hooks/useDetectionStream';
import { useAlertState }       from '../hooks/useAlertState';
import Header                  from '../components/layout/Header';
import VideoCanvas             from '../components/monitor/VideoCanvas';
import ThreatLogTable          from '../components/alerts/ThreatLogTable';
import RedAlertBanner          from '../components/alerts/RedAlertBanner';

// Recent-events ring buffer — keeps last N events for the side panel
function useRecentEvents(events, maxLen = 30) {
  const [log, setLog] = useState([]);
  React.useEffect(() => {
    if (!events || events.length === 0) return;
    setLog(prev => [...events, ...prev].slice(0, maxLen));
  }, [events, maxLen]);
  return log;
}

const CAMERA_ID = import.meta.env.VITE_CAMERA_ID || 'CAM-01';

export default function LiveMonitor() {
  const { lastMessage, isConnected } = useWebSocket(CAMERA_ID);
  const { frame, detections, threatEvents, threatLevel, gradcamB64 } =
    useDetectionStream(lastMessage);
  const { isAlertActive, activeEvent, dismissAlert } =
    useAlertState(threatLevel, threatEvents);
  const recentEvents = useRecentEvents(threatEvents);

  const unacknowledgedCount = recentEvents.filter(
    e => e.threat_level === 'HIGH' && !e.acknowledged
  ).length;

  return (
    <div className="main-content">
      <Header
        title="Live Surveillance"
        isConnected={isConnected}
        alertCount={unacknowledgedCount}
      />

      <div className="page" style={{ paddingBottom: '1rem' }}>
        <div className="monitor-layout">
          {/* Main canvas */}
          <VideoCanvas
            frame={frame}
            detections={detections}
            gradcamB64={gradcamB64}
            threatLevel={threatLevel}
            cameraId={CAMERA_ID}
          />

          {/* Side panel — recent events */}
          <div className="side-panel">
            <div className="side-panel__header">
              Recent Detections
              <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
                {recentEvents.length}
              </span>
            </div>
            <div className="side-panel__body">
              <ThreatLogTable
                events={recentEvents}
                compact={true}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Red Alert overlay — renders outside page flow */}
      <RedAlertBanner
        isActive={isAlertActive}
        event={activeEvent}
        onDismiss={dismissAlert}
      />
    </div>
  );
}
