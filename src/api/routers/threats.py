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

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/threats", tags=["threats"])


@router.get("")
async def list_threats(
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
    ...


@router.get("/{event_id}")
async def get_threat(event_id: str):
    """
    Returns a single ThreatEvent by ID.
    Includes gradcam_b64 field if a Grad-CAM image was generated.
    """
    ...


@router.patch("/{event_id}/ack")
async def acknowledge_threat(event_id: str):
    """
    Mark a threat event as acknowledged.
    Called by React RedAlertBanner dismiss button.
    """
    ...


@router.delete("/{event_id}")
async def delete_threat(event_id: str):
    """
    Permanently delete a threat event record.
    Admin operation — requires API key header validation.
    """
    ...
