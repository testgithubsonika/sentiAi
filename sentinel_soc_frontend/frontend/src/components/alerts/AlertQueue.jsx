/**
 * components/alerts/AlertQueue.jsx
 * ===================================
 * Real-time, sortable alert table. New alerts arrive prepended (via
 * AnomalyStreamProvider) for a live "newest first" feel; a sort toggle
 * lets an analyst switch to strict risk-score ranking for triage. The
 * newest row plays a brief highlight/entrance animation so a live alert
 * is noticeable without needing a toast or sound.
 */

import { useMemo, useState } from 'react';
import { ChevronRight, ArrowDownUp } from 'lucide-react';
import { SeverityBadge, CategoryBadge } from './SeverityBadge';
import { severityFromRisk, formatScore, formatTime, formatRelative } from '../../constants';

export function AlertQueue({ alerts, onSelectAlert }) {
  const [sortBy, setSortBy] = useState('newest'); // 'newest' | 'risk'

  const sorted = useMemo(() => {
    if (sortBy === 'risk') {
      return [...alerts].sort((a, b) => (b.risk_score ?? 0) - (a.risk_score ?? 0));
    }
    return alerts; // already newest-first (prepended on arrival / sort_by=risk_score on initial seed)
  }, [alerts, sortBy]);

  return (
    <section className="rounded-md border border-hairline bg-panel shadow-panel">
      <header className="flex items-center justify-between border-b border-hairline px-4 py-3">
        <div>
          <h2 className="font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
            Alert Queue
          </h2>
          <p className="mt-0.5 text-xs text-ink-faint">{alerts.length} flagged anomalies</p>
        </div>
        <button
          type="button"
          onClick={() => setSortBy((s) => (s === 'newest' ? 'risk' : 'newest'))}
          className="flex items-center gap-1.5 rounded border border-hairline bg-panel-raised px-2.5 py-1.5 font-mono text-2xs text-ink-dim transition-colors hover:text-ink"
        >
          <ArrowDownUp className="h-3 w-3" />
          {sortBy === 'newest' ? 'Newest' : 'Highest Risk'}
        </button>
      </header>

      <div className="scroll-thin max-h-[520px] overflow-y-auto">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 z-10 bg-panel">
            <tr className="border-b border-hairline text-2xs uppercase tracking-wide text-ink-faint">
              <th className="px-4 py-2 font-medium">Entity</th>
              <th className="px-4 py-2 font-medium">Category</th>
              <th className="px-4 py-2 font-medium">Severity</th>
              <th className="px-4 py-2 font-medium">Risk</th>
              <th className="px-4 py-2 font-medium">Detected</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((alert, i) => (
              <tr
                key={alert.alert_id}
                className={`border-b border-hairline/60 transition-colors hover:bg-panel-raised/60 ${
                  i === 0 && sortBy === 'newest' ? 'animate-row-in' : ''
                }`}
              >
                <td className="px-4 py-2.5 font-mono text-xs text-ink">{alert.entity_id}</td>
                <td className="px-4 py-2.5">
                  <CategoryBadge category={alert.alert_type} />
                </td>
                <td className="px-4 py-2.5">
                  <SeverityBadge severity={alert.severity ?? severityFromRisk(alert.risk_score ?? 0)} />
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-ink-dim">{formatScore(alert.risk_score)}</td>
                <td className="px-4 py-2.5 font-mono text-2xs text-ink-faint" title={formatTime(alert.timestamp)}>
                  {formatRelative(alert.timestamp)}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <button
                    type="button"
                    onClick={() => onSelectAlert?.(alert)}
                    className="inline-flex items-center gap-0.5 rounded px-2 py-1 font-mono text-2xs text-signal transition-colors hover:bg-signal/10"
                  >
                    Details <ChevronRight className="h-3 w-3" />
                  </button>
                </td>
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-xs text-ink-faint">
                  No anomalies flagged yet — the queue populates as the stream runs.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
