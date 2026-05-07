/**
 * ui/src/pages/ThreatHistory.jsx
 *
 * Historical threat event log with pagination and filtering.
 * Fetches from GET /api/threats — does NOT use the WebSocket stream.
 */
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Header          from '../components/layout/Header';
import ThreatLogTable  from '../components/alerts/ThreatLogTable';
import { Filter, RefreshCw } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const PAGE_SIZE = 50;

export default function ThreatHistory() {
  const [events,    setEvents]    = useState([]);
  const [page,      setPage]      = useState(1);
  const [hasMore,   setHasMore]   = useState(true);
  const [loading,   setLoading]   = useState(false);
  const [filters,   setFilters]   = useState({ level: '', class_name: '' });

  const fetchEvents = useCallback(async (pageNum = 1, replace = false) => {
    setLoading(true);
    try {
      const params = {
        page: pageNum, page_size: PAGE_SIZE,
        ...(filters.level      && { level:      filters.level }),
        ...(filters.class_name && { class_name: filters.class_name }),
      };
      const { data } = await axios.get(`${API_BASE}/api/threats`, { params });
      setEvents(prev => replace ? data : [...prev, ...data]);
      setHasMore(data.length === PAGE_SIZE);
    } catch {
      // network error — show existing data
    } finally {
      setLoading(false);
    }
  }, [filters]);

  // Re-fetch from page 1 when filters change
  useEffect(() => {
    setPage(1);
    fetchEvents(1, true);
  }, [fetchEvents]);

  const loadMore = () => {
    const next = page + 1;
    setPage(next);
    fetchEvents(next, false);
  };

  return (
    <div className="main-content">
      <Header title="Threat Log" isConnected={false} />

      <div className="page">
        {/* Filter bar */}
        <div className="flex-between gap-3" style={{ flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Filter size={16} style={{ color: 'var(--text-muted)' }} />
            <select
              value={filters.level}
              onChange={e => setFilters(f => ({ ...f, level: e.target.value }))}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-normal)',
                borderRadius: '8px', color: 'var(--text-secondary)',
                padding: '0.4rem 0.75rem', fontSize: '0.8rem', cursor: 'pointer',
              }}
              aria-label="Filter by threat level"
            >
              <option value="">All Levels</option>
              <option value="HIGH">HIGH</option>
              <option value="LOW">LOW</option>
            </select>

            <select
              value={filters.class_name}
              onChange={e => setFilters(f => ({ ...f, class_name: e.target.value }))}
              style={{
                background: 'var(--bg-card)', border: '1px solid var(--border-normal)',
                borderRadius: '8px', color: 'var(--text-secondary)',
                padding: '0.4rem 0.75rem', fontSize: '0.8rem', cursor: 'pointer',
              }}
              aria-label="Filter by weapon class"
            >
              <option value="">All Classes</option>
              <option value="handgun">Handgun</option>
              <option value="rifle">Rifle</option>
              <option value="knife">Knife</option>
              <option value="machete">Machete</option>
            </select>
          </div>

          <button
            onClick={() => fetchEvents(1, true)}
            style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              background: 'var(--bg-card)', border: '1px solid var(--border-normal)',
              borderRadius: '8px', color: 'var(--text-secondary)',
              padding: '0.4rem 0.75rem', fontSize: '0.8rem', cursor: 'pointer',
            }}
            aria-label="Refresh threat log"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            Refresh
          </button>
        </div>

        {/* Table */}
        <ThreatLogTable events={events} compact={false} />

        {/* Load more */}
        {hasMore && (
          <div className="flex-center">
            <button
              onClick={loadMore}
              disabled={loading}
              className="save-btn"
              style={{ background: 'var(--bg-card)', boxShadow: 'none',
                       border: '1px solid var(--border-normal)', color: 'var(--text-secondary)' }}
            >
              {loading ? 'Loading…' : 'Load More'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
