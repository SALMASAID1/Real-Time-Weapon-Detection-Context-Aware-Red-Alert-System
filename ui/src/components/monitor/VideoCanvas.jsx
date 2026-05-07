/**
 * ui/src/components/monitor/VideoCanvas.jsx
 *
 * Renders the live surveillance video stream using an HTML5 <canvas> element.
 *
 * WHY <canvas> and NOT <img src="data:..."> refreshing on each frame:
 *   - <img> triggers a full browser layout recalculation on src change at 30 FPS.
 *   - <canvas> ctx.drawImage() is GPU-composited — the browser never reflows.
 *   - Bounding boxes and labels are drawn in a second pass on the same canvas
 *     without touching the DOM at all.
 *
 * Frame rendering pipeline (per WebSocket message):
 *   1. Receive base64 JPEG string (frame.frameB64)
 *   2. Create an Image object, set .src = "data:image/jpeg;base64,<frame>"
 *   3. On image.onload: ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
 *   4. For each detection in detections[]:
 *      a. ctx.strokeRect(x1, y1, w, h)  — bounding box
 *      b. ctx.fillText(className + confidence%) — label chip
 *      c. Color: weapon classes → red, hand → cyan, person → blue
 *   5. If gradcamB64 is non-null and showGradcam is true:
 *      draw a semi-transparent Grad-CAM layer over the frame.
 *
 * Props:
 *   frame       : { frameB64, fps }  — from useDetectionStream
 *   detections  : Detection[]
 *   gradcamB64  : string|null
 *   showGradcam : boolean
 *   threatLevel : string
 *   cameraId    : string
 */
import React, { useRef, useEffect, useState } from 'react';
import { VideoOff, Eye, EyeOff } from 'lucide-react';
import ThreatBadge from './ThreatBadge';

// Color map per class category
const BBOX_COLORS = {
  weapon:  '#ef4444',
  hand:    '#06b6d4',
  person:  '#3b82f6',
  default: '#f59e0b',
};

function classColor(detection) {
  if (detection.is_weapon)                   return BBOX_COLORS.weapon;
  if (detection.class_name === 'hand')       return BBOX_COLORS.hand;
  if (detection.class_name === 'person')     return BBOX_COLORS.person;
  return BBOX_COLORS.default;
}

export default function VideoCanvas({
  frame, detections = [], gradcamB64 = null,
  showGradcam = true, threatLevel = 'NONE', cameraId = 'CAM-01',
}) {
  const canvasRef    = useRef(null);
  const gradcamRef   = useRef(null);
  const [gcam, setGcam] = useState(true); // local toggle

  // Draw frame + bboxes whenever a new frame arrives
  useEffect(() => {
    if (!frame?.frameB64) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = () => {
      // Resize canvas to match natural image dimensions once
      if (canvas.width !== img.naturalWidth) {
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }

      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes
      detections.forEach(det => {
        const { x1, y1, x2, y2 } = det.bbox;
        const w = x2 - x1, h = y2 - y1;
        const color = classColor(det);
        const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`;

        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.strokeRect(x1, y1, w, h);

        // Label chip background
        ctx.fillStyle = color;
        const textW = ctx.measureText(label).width + 12;
        ctx.fillRect(x1, y1 - 20, textW, 20);

        // Label text
        ctx.fillStyle    = '#fff';
        ctx.font         = '600 12px Inter, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x1 + 6, y1 - 10);
      });

      // Grad-CAM overlay layer
      if (gcam && showGradcam && gradcamB64) {
        const heatmap = new Image();
        heatmap.onload = () => {
          ctx.globalAlpha = 0.45;
          ctx.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
          ctx.globalAlpha = 1.0;
        };
        heatmap.src = `data:image/jpeg;base64,${gradcamB64}`;
      }
    };
    img.src = `data:image/jpeg;base64,${frame.frameB64}`;
  }, [frame, detections, gradcamB64, gcam, showGradcam]);

  const isAlert = threatLevel === 'HIGH';

  return (
    <div className={`video-container${isAlert ? ' video-container--alert' : ''}`}>
      {frame?.frameB64 ? (
        <>
          <canvas ref={canvasRef} className="video-canvas" />
          <div className="video-overlay">
            <div className="video-overlay__top">
              <span className="camera-chip">{cameraId}</span>
              <ThreatBadge level={threatLevel} />
            </div>
            <div className="video-overlay__bottom">
              <span className="fps-counter">
                {frame.fps ? `${frame.fps.toFixed(1)} FPS` : '--'}
              </span>
              {gradcamB64 && (
                <button
                  onClick={() => setGcam(v => !v)}
                  style={{
                    background: 'rgba(0,0,0,.65)', backdropFilter: 'blur(8px)',
                    border: '1px solid var(--border-normal)', borderRadius: '6px',
                    padding: '0.2rem 0.6rem', color: 'var(--text-secondary)',
                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                    fontSize: '0.7rem', cursor: 'pointer',
                  }}
                  title="Toggle Grad-CAM overlay"
                >
                  {gcam ? <EyeOff size={12}/> : <Eye size={12}/>}
                  Grad-CAM
                </button>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="video-placeholder">
          <VideoOff size={48} className="video-placeholder__icon" />
          <span className="video-placeholder__text">Waiting for backend stream…</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {cameraId} · Connecting to ws://localhost:8000
          </span>
        </div>
      )}
    </div>
  );
}
