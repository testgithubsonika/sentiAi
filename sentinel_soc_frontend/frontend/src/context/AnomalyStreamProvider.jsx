/**
 * context/AnomalyStreamProvider.jsx
 * ===================================
 * Owns the single `/ws/stream` connection plus the REST-seeded initial
 * state, and derives everything the dashboard renders from it: the
 * ranked alert queue, a rolling event buffer (for the pulse strip + the
 * request-volume chart), and live metrics. Every chart/list component
 * reads from this context instead of managing its own fetch/socket, so
 * a new alert updates the queue, the charts, and the header counters in
 * one render pass -- no polling, no page refresh.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { api, getStreamUrl } from '../api/client';
import { useWebSocketStream } from '../hooks/useWebSocketStream';

const AnomalyStreamContext = createContext(null);

const MAX_ALERTS = 250;
const MAX_EVENT_BUFFER = 400;

export function AnomalyStreamProvider({ children }) {
  const [alerts, setAlerts] = useState([]);
  const [events, setEvents] = useState([]); // rolling buffer of {ts, kind: 'normal'|'anomalous'}
  const [metrics, setMetrics] = useState(null);
  const [loadingSeed, setLoadingSeed] = useState(true);
  const throughputRef = useRef({ windowStart: Date.now(), count: 0 });
  const [eventsPerSecond, setEventsPerSecond] = useState(0);

  // -- seed from REST on mount -------------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [alertRes, metricsRes] = await Promise.all([
          api.listAlerts({ limit: 100, sort_by: 'risk_score' }),
          api.getMetrics(),
        ]);
        if (cancelled) return;
        setAlerts(alertRes.alerts ?? []);
        setMetrics(metricsRes);
      } catch (err) {
        console.error('Failed to seed dashboard from REST API', err);
      } finally {
        if (!cancelled) setLoadingSeed(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // -- periodic metrics refresh (cheap, backstops any dropped WS frames) --
  useEffect(() => {
    const interval = setInterval(() => {
      api.getMetrics().then(setMetrics).catch(() => {});
    }, 15_000);
    return () => clearInterval(interval);
  }, []);

  const recordThroughputTick = useCallback(() => {
    const now = Date.now();
    const bucket = throughputRef.current;
    bucket.count += 1;
    const elapsed = (now - bucket.windowStart) / 1000;
    if (elapsed >= 2) {
      setEventsPerSecond(bucket.count / elapsed);
      throughputRef.current = { windowStart: now, count: 0 };
    }
  }, []);

  const handleMessage = useCallback(
    (payload) => {
      recordThroughputTick();
      const ts = payload.timestamp ? new Date(payload.timestamp) : new Date();

      if (payload.type === 'alert') {
        setAlerts((prev) => [payload, ...prev].slice(0, MAX_ALERTS));
        setEvents((prev) => [...prev, { ts, kind: 'anomalous' }].slice(-MAX_EVENT_BUFFER));
        setMetrics((prev) =>
          prev
            ? {
                ...prev,
                total_alerts: prev.total_alerts + 1,
                active_anomalies: prev.active_anomalies + 1,
                alerts_by_type: {
                  ...prev.alerts_by_type,
                  [payload.alert_type]: (prev.alerts_by_type?.[payload.alert_type] ?? 0) + 1,
                },
              }
            : prev
        );
      } else if (payload.type === 'event') {
        setEvents((prev) => [...prev, { ts, kind: 'normal' }].slice(-MAX_EVENT_BUFFER));
        setMetrics((prev) => (prev ? { ...prev, total_events_processed: prev.total_events_processed + 1 } : prev));
      }
    },
    [recordThroughputTick]
  );

  const { status } = useWebSocketStream(getStreamUrl(), { onMessage: handleMessage });

  const acknowledgeAlert = useCallback(async (alertId, status_) => {
    const updated = await api.updateAlertStatus(alertId, { status: status_ });
    setAlerts((prev) => prev.map((a) => (a.alert_id === alertId ? { ...a, status: updated.status } : a)));
  }, []);

  const value = useMemo(
    () => ({
      alerts,
      events,
      metrics,
      connectionStatus: status,
      eventsPerSecond,
      loadingSeed,
      acknowledgeAlert,
    }),
    [alerts, events, metrics, status, eventsPerSecond, loadingSeed, acknowledgeAlert]
  );

  return <AnomalyStreamContext.Provider value={value}>{children}</AnomalyStreamContext.Provider>;
}

export function useAnomalyStream() {
  const ctx = useContext(AnomalyStreamContext);
  if (!ctx) throw new Error('useAnomalyStream must be used within an AnomalyStreamProvider');
  return ctx;
}
