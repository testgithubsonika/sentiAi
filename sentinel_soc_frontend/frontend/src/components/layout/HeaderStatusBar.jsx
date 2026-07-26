/**
 * components/layout/HeaderStatusBar.jsx
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, Gauge } from 'lucide-react';

const STATUS_META = {
  open: { label: 'LIVE', dotClass: 'bg-signal animate-pulse-dot', textClass: 'text-signal' },
  connecting: { label: 'CONNECTING', dotClass: 'bg-sev-medium animate-pulse-dot', textClass: 'text-sev-medium' },
  closed: { label: 'RECONNECTING', dotClass: 'bg-sev-high animate-pulse-dot', textClass: 'text-sev-high' },
  error: { label: 'CONNECTION ERROR', dotClass: 'bg-sev-critical animate-pulse-dot', textClass: 'text-sev-critical' },
};

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export function HeaderStatusBar({ connectionStatus, eventsPerSecond, activeAnomalies }) {
  const now = useClock();
  const meta = STATUS_META[connectionStatus] ?? STATUS_META.connecting;

  return (
    <header className="flex h-14 items-center justify-between border-b border-hairline bg-panel px-5">
      <div className="flex items-center gap-2">
        <h1 className="text-sm font-semibold text-ink">Behavioral Anomaly Detection</h1>
        <span className="hidden font-mono text-2xs text-ink-faint sm:inline">/ SOC Overview</span>
      </div>

      <div className="flex items-center gap-5">
        <div className="hidden items-center gap-1.5 font-mono text-2xs text-ink-dim sm:flex">
          <Gauge className="h-3.5 w-3.5" />
          {eventsPerSecond.toFixed(1)} evt/s
        </div>

        <div className="flex items-center gap-1.5 font-mono text-2xs text-ink-dim">
          <AlertTriangle className="h-3.5 w-3.5 text-sev-high" />
          <span className="text-ink">{activeAnomalies ?? '—'}</span> active
        </div>

        <div className="hidden font-mono text-2xs text-ink-faint md:block">
          {now.toLocaleTimeString([], { hour12: false })}
        </div>

        <div className={`flex items-center gap-2 rounded-full border border-hairline bg-void px-2.5 py-1 font-mono text-2xs font-medium ${meta.textClass}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${meta.dotClass}`} />
          {meta.label}
        </div>
      </div>
    </header>
  );
}
