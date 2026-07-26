/**
 * components/entity/EntityHistoryPanel.jsx
 * ============================================
 * Slide-over panel opened from "Details" in the alert queue. Fetches
 * `GET /api/entities/{id}/history` (entity profile + recent logs +
 * the live River adaptive baseline) and layers the *selected alert's*
 * SHAP-attributed raw values on top of that baseline so an analyst can
 * see, at a glance, which features actually deviated and by how much.
 */

import { useEffect, useState } from 'react';
import { X, MapPin, Clock, Server } from 'lucide-react';
import { api } from '../../api/client';
import { ContributingFactors } from '../explainability/ContributingFactors';
import { formatTime, formatScore } from '../../constants';
import { SeverityBadge } from '../alerts/SeverityBadge';

const BASELINE_LABELS = {
  login_hour: 'Login Hour',
  session_duration: 'Session Duration (s)',
  geo_distance_km: 'Geo Distance (km)',
  failure_count: 'Auth Failures',
  num_commands: 'Command Count',
};

function BaselineRow({ label, mean, std, current }) {
  const deviation = std > 0 && current != null ? Math.abs(current - mean) / std : 0;
  const tone = deviation >= 2 ? 'var(--sev-critical)' : deviation >= 1 ? 'var(--sev-high)' : 'var(--accent-signal)';

  // position the baseline range + current marker on a shared 0-100 scale
  const lo = mean - std * 2;
  const hi = mean + std * 2;
  const span = Math.max(0.0001, hi - lo);
  const clamp = (v) => Math.min(100, Math.max(0, ((v - lo) / span) * 100));

  return (
    <div className="mb-3">
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-ink-dim">{label}</span>
        <span className="font-mono text-2xs text-ink-faint">
          baseline {mean.toFixed(1)} ± {std.toFixed(1)}
          {current != null && (
            <span className="ml-2 font-semibold" style={{ color: tone }}>
              now {current.toFixed(1)}
            </span>
          )}
        </span>
      </div>
      <div className="relative h-2 rounded-full bg-panel-raised">
        <div
          className="absolute h-full rounded-full bg-signal/20"
          style={{ left: `${clamp(mean - std)}%`, width: `${clamp(mean + std) - clamp(mean - std)}%` }}
        />
        <div className="absolute h-full w-px bg-ink-faint/50" style={{ left: `${clamp(mean)}%` }} />
        {current != null && (
          <div
            className="absolute -top-0.5 h-3 w-1 rounded-full"
            style={{ left: `${clamp(current)}%`, backgroundColor: tone }}
          />
        )}
      </div>
    </div>
  );
}

export function EntityHistoryPanel({ alert, onClose }) {
  const [history, setHistory] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | loading | error | ready

  useEffect(() => {
    if (!alert) return;
    setStatus('loading');
    api
      .getEntityHistory(alert.entity_id, { log_limit: 20, alert_limit: 10 })
      .then((res) => {
        setHistory(res);
        setStatus('ready');
      })
      .catch(() => setStatus('error'));
  }, [alert]);

  const open = Boolean(alert);

  return (
    <div
      className={`fixed inset-0 z-40 ${open ? 'pointer-events-auto' : 'pointer-events-none'}`}
      aria-hidden={!open}
    >
      <div
        className={`absolute inset-0 bg-void/70 transition-opacity ${open ? 'opacity-100' : 'opacity-0'}`}
        onClick={onClose}
      />

      <aside
        className={`absolute right-0 top-0 h-full w-full max-w-md border-l border-hairline bg-panel shadow-panel ${
          open ? 'animate-slide-in' : 'translate-x-full'
        }`}
      >
        {alert && (
          <div className="flex h-full flex-col">
            <header className="flex items-start justify-between border-b border-hairline px-5 py-4">
              <div>
                <p className="font-mono text-2xs uppercase tracking-widest text-ink-faint">Entity Drill-down</p>
                <h2 className="mt-1 font-mono text-base font-semibold text-ink">{alert.entity_id}</h2>
                <div className="mt-2 flex items-center gap-2">
                  <SeverityBadge severity={alert.severity} />
                  <span className="font-mono text-2xs text-ink-faint">risk {formatScore(alert.risk_score)}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded p-1.5 text-ink-faint transition-colors hover:bg-panel-raised hover:text-ink"
                aria-label="Close panel"
              >
                <X className="h-4 w-4" />
              </button>
            </header>

            <div className="scroll-thin flex-1 overflow-y-auto px-5 py-4">
              {status === 'loading' && <p className="text-xs text-ink-faint">Loading entity history…</p>}
              {status === 'error' && (
                <p className="text-xs text-sev-high">Couldn't load history for this entity.</p>
              )}

              {status === 'ready' && history && (
                <>
                  <section className="mb-6">
                    <h3 className="mb-2 font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
                      Profile
                    </h3>
                    <div className="space-y-1.5 rounded border border-hairline bg-panel-raised/40 p-3 text-xs text-ink-dim">
                      <p className="flex items-center gap-2">
                        <Server className="h-3.5 w-3.5 text-ink-faint" />
                        {history.entity.display_name} &middot; {history.entity.entity_type}
                      </p>
                      <p className="flex items-center gap-2">
                        <MapPin className="h-3.5 w-3.5 text-ink-faint" />
                        {history.entity.home_city}, {history.entity.home_country}
                      </p>
                      <p className="flex items-center gap-2">
                        <Clock className="h-3.5 w-3.5 text-ink-faint" />
                        Habitual hours {history.entity.habitual_hour_start}:00–{history.entity.habitual_hour_end}:00
                      </p>
                    </div>
                  </section>

                  <section className="mb-6">
                    <h3 className="mb-3 font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
                      Baseline vs. This Session
                    </h3>
                    {history.adaptive_baseline ? (
                      Object.entries(history.adaptive_baseline).map(([feature, { mean, std }]) => {
                        const factor = (alert.contributing_factors ?? []).find((f) => f.feature === feature);
                        return (
                          <BaselineRow
                            key={feature}
                            label={BASELINE_LABELS[feature] ?? feature}
                            mean={mean}
                            std={std}
                            current={factor?.raw_value}
                          />
                        );
                      })
                    ) : (
                      <p className="text-xs text-ink-faint">
                        No live baseline yet for this entity (worker restarted or entity is cold-start).
                      </p>
                    )}
                  </section>

                  <section className="mb-6">
                    <h3 className="mb-2 font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
                      Why This Was Flagged
                    </h3>
                    <ContributingFactors summary={alert.summary} factors={alert.contributing_factors} />
                  </section>

                  <section>
                    <h3 className="mb-2 font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
                      Recent Activity
                    </h3>
                    <ul className="scroll-thin max-h-48 space-y-1.5 overflow-y-auto">
                      {history.recent_logs.map((log) => (
                        <li
                          key={log.log_id}
                          className="flex items-center justify-between rounded border border-hairline/60 px-2.5 py-1.5 text-2xs"
                        >
                          <span className="font-mono text-ink-faint">{formatTime(log.timestamp)}</span>
                          <span className="truncate text-ink-dim">{log.resource_accessed || '—'}</span>
                          <span
                            className={`font-mono ${log.label === 'normal' ? 'text-ink-faint' : 'text-sev-high'}`}
                          >
                            {log.label}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </section>
                </>
              )}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
