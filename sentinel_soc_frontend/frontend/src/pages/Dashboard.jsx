/**
 * pages/Dashboard.jsx
 * ======================
 * The "Overview" screen: three analytics charts + the live alert queue,
 * all reading from the shared AnomalyStreamProvider context. Selecting
 * an alert opens the EntityHistoryPanel drill-down.
 */

import { useState } from 'react';
import { RequestVolumeChart } from '../components/charts/RequestVolumeChart';
import { AnomalyByCategoryChart } from '../components/charts/AnomalyByCategoryChart';
import { RiskScoreDistributionChart } from '../components/charts/RiskScoreDistributionChart';
import { AlertQueue } from '../components/alerts/AlertQueue';
import { EntityHistoryPanel } from '../components/entity/EntityHistoryPanel';
import { useAnomalyStream } from '../context/AnomalyStreamProvider';

export function Dashboard() {
  const { alerts, events, metrics, loadingSeed } = useAnomalyStream();
  const [selectedAlert, setSelectedAlert] = useState(null);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <RequestVolumeChart events={events} />
        <AnomalyByCategoryChart alertsByType={metrics?.alerts_by_type} />
        <RiskScoreDistributionChart alerts={alerts} />
      </div>

      {loadingSeed ? (
        <div className="rounded-md border border-hairline bg-panel p-10 text-center text-xs text-ink-faint shadow-panel">
          Loading alert history…
        </div>
      ) : (
        <AlertQueue alerts={alerts} onSelectAlert={setSelectedAlert} />
      )}

      <EntityHistoryPanel alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
    </div>
  );
}
