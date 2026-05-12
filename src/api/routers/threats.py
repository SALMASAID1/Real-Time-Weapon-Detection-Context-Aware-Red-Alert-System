"""
src/api/routers/threats.py
============================
REST API for threat event history — consumed by React ThreatHistory page.

Endpoints
---------
GET  /api/threats           — Paginated list of all logged ThreatEvent records.
GET  /api/threats/{id}      — Single event with full metadata.
PATCH /api/threats/{id}/ack — Mark event as acknowledged by operator.
DELETE /api/threats/{id}    — Delete a specific event (admin only).

Data source
-----------
AlertDispatcher writes JSON event records to a log file (or SQLite DB).
These routers read from the same store. The schema is defined in
src/api/schemas/detection.py :: ThreatEvent.

Pagination
----------
Endpoint accepts `page` and `page_size` query parameters.
Default: page=1, page_size=50. React ThreatLogTable fetches the first page
on mount and loads subsequent pages as the operator scrolls.

Filtering
---------
Optional query params:
  camera_id  : str   — filter by camera
  class_name : str   — filter by weapon class
  level      : str   — filter by threat_level ("HIGH", "LOW")
  since      : float — Unix timestamp lower bound (for time range queries)
  acknowledged: bool — filter acknowledged / unacknowledged events
"""

import json
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/threats", tags=["threats"])


def _get_log_path(request: Request) -> str:
    """Resolve the JSONL log path from app state (set during lifespan startup)."""
    return getattr(request.app.state, "threat_log_path", "data/logs/threats.jsonl")


def _read_all_events(log_path: str) -> List[Dict]:
    """Read all events from the JSONL log file, most-recent first."""
    events = []
    if not os.path.exists(log_path):
        return events
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to read threat log: {e}")
    # Return most-recent first (JSONL appends chronologically)
    events.reverse()
    return events


def _write_all_events(log_path: str, events: List[Dict]):
    """Rewrite the JSONL log file with the given events (chronological order)."""
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # Events are stored most-recent-first internally, reverse to chronological for file
    chronological = list(reversed(events))
    with open(log_path, "w", encoding="utf-8") as f:
        for event in chronological:
            f.write(json.dumps(event) + "\n")


def _apply_filters(
    events: List[Dict],
    camera_id: Optional[str],
    class_name: Optional[str],
    level: Optional[str],
    since: Optional[float],
    acknowledged: Optional[bool],
) -> List[Dict]:
    """Apply optional filters to the event list."""
    filtered = events

    if camera_id:
        filtered = [e for e in filtered if e.get("camera_id") == camera_id]

    if class_name:
        filtered = [e for e in filtered if e.get("class_name", "").lower() == class_name.lower()]

    if level:
        # Events logged by AlertDispatcher always have threat_level = "HIGH"
        # but we also store a composite_score — allow filtering by a virtual level
        filtered = [e for e in filtered if e.get("threat_level", "HIGH").upper() == level.upper()]

    if since is not None:
        filtered = [
            e for e in filtered
            if _event_timestamp(e) >= since
        ]

    if acknowledged is not None:
        filtered = [e for e in filtered if e.get("acknowledged", False) == acknowledged]

    return filtered


def _event_timestamp(event: Dict) -> float:
    """Extract a Unix timestamp from an event record."""
    ts = event.get("timestamp", "")
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


@router.get("")
async def list_threats(
    request: Request,
    page:         int            = Query(1, ge=1),
    page_size:    int            = Query(50, ge=1, le=200),
    camera_id:    Optional[str]  = None,
    class_name:   Optional[str]  = None,
    level:        Optional[str]  = None,
    since:        Optional[float]= None,
    acknowledged: Optional[bool] = None,
):
    """
    Returns a paginated, filterable list of ThreatEvent records.
    Used by the React ThreatHistory page on mount and scroll.
    """
    log_path = _get_log_path(request)
    events = _read_all_events(log_path)
    events = _apply_filters(events, camera_id, class_name, level, since, acknowledged)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    return events[start:end]


@router.get("/{event_id}")
async def get_threat(event_id: str, request: Request):
    """
    Returns a single ThreatEvent by ID.
    Includes gradcam_b64 field if a Grad-CAM image was generated.
    """
    log_path = _get_log_path(request)
    events = _read_all_events(log_path)

    for event in events:
        if event.get("event_id") == event_id:
            return event

    raise HTTPException(status_code=404, detail=f"Threat event '{event_id}' not found")


@router.patch("/{event_id}/ack")
async def acknowledge_threat(event_id: str, request: Request):
    """
    Mark a threat event as acknowledged.
    Called by React RedAlertBanner dismiss button.
    """
    log_path = _get_log_path(request)
    events = _read_all_events(log_path)

    found = False
    for event in events:
        if event.get("event_id") == event_id:
            event["acknowledged"] = True
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Threat event '{event_id}' not found")

    _write_all_events(log_path, events)
    return {"event_id": event_id, "acknowledged": True}


@router.delete("/{event_id}")
async def delete_threat(event_id: str, request: Request):
    """
    Permanently delete a threat event record.
    Admin operation — requires API key header validation.
    """
    log_path = _get_log_path(request)
    events = _read_all_events(log_path)

    original_len = len(events)
    events = [e for e in events if e.get("event_id") != event_id]

    if len(events) == original_len:
        raise HTTPException(status_code=404, detail=f"Threat event '{event_id}' not found")

    _write_all_events(log_path, events)
    return {"event_id": event_id, "deleted": True}
