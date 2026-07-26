/**
 * components/layout/LivePulseStrip.jsx
 * =======================================
 * The dashboard's signature element: a thin telemetry ticker rendering
 * the last ~80 stream events as vertical bars, teal for normal traffic
 * and severity-colored for alerts, newest on the right. It's the one
 * place the raw, ungrouped event stream is visible at a glance -- a
 * literal pulse for the system the rest of the dashboard summarizes.
 */

import { useMemo } from 'react';

const BAR_COUNT = 80;

export function LivePulseStrip({ events }) {
  const bars = useMemo(() => {
    const recent = events.slice(-BAR_COUNT);
    const padding = Array.from({ length: Math.max(0, BAR_COUNT - recent.length) }, () => null);
    return [...padding, ...recent];
  }, [events]);

  return (
    <div className="flex h-9 items-end gap-[3px] overflow-hidden border-b border-hairline bg-void px-5 py-2">
      {bars.map((event, i) => {
        if (!event) {
          return <span key={i} className="h-1 w-full rounded-sm bg-hairline/60" />;
        }
        const isAlert = event.kind === 'anomalous';
        const height = isAlert ? '100%' : `${28 + ((i * 37) % 30)}%`; // gentle jitter for a live, non-static feel
        const color = isAlert ? 'var(--sev-high)' : 'var(--accent-signal-dim)';
        return (
          <span
            key={i}
            className="w-full rounded-sm transition-all duration-300"
            style={{ height, backgroundColor: color, opacity: isAlert ? 1 : 0.8 }}
            title={isAlert ? 'Anomalous event' : 'Normal event'}
          />
        );
      })}
    </div>
  );
}
