/**
 * components/layout/AppShell.jsx
 * =================================
 * Top-level dashboard layout. Reads live connection/throughput state
 * from AnomalyStreamProvider so the header + pulse strip update without
 * any prop drilling from the page.
 */

import { Sidebar } from './Sidebar';
import { HeaderStatusBar } from './HeaderStatusBar';
import { LivePulseStrip } from './LivePulseStrip';
import { useAnomalyStream } from '../../context/AnomalyStreamProvider';

export function AppShell({ children, activeNavId, onNavigate }) {
  const { connectionStatus, eventsPerSecond, events, metrics } = useAnomalyStream();

  return (
    <div className="flex h-screen bg-void text-ink">
      <Sidebar activeId={activeNavId} onNavigate={onNavigate} />

      <div className="flex min-w-0 flex-1 flex-col">
        <HeaderStatusBar
          connectionStatus={connectionStatus}
          eventsPerSecond={eventsPerSecond}
          activeAnomalies={metrics?.active_anomalies}
        />
        <LivePulseStrip events={events} />

        <main className="scroll-thin flex-1 overflow-y-auto bg-grid px-6 py-6">
          <div className="mx-auto max-w-[1400px]">{children}</div>
        </main>
      </div>
    </div>
  );
}
