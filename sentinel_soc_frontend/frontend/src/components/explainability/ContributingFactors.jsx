/**
 * components/explainability/ContributingFactors.jsx
 * =====================================================
 * Renders `Alert.contributing_factors` (from explainability.py's
 * `combine_and_summarize`) as ranked bars: each factor's bar width is
 * its |shap_value| normalized against the strongest factor in the list,
 * so the analyst sees relative weight at a glance. A small source pill
 * (IFOREST / XGBOOST) shows which model layer surfaced it.
 */

const SOURCE_META = {
  iforest: { label: 'IFOREST', className: 'text-signal border-signal/30 bg-signal/10' },
  xgboost: { label: 'XGBOOST', className: 'text-sev-medium border-sev-medium/30 bg-sev-medium/10' },
};

export function ContributingFactors({ summary, factors = [] }) {
  const maxAbs = Math.max(0.0001, ...factors.map((f) => Math.abs(f.shap_value ?? 0)));

  return (
    <div>
      {summary && (
        <p className="mb-3 rounded border border-hairline bg-panel-raised px-3 py-2 text-sm text-ink">
          {summary}
        </p>
      )}

      <ul className="space-y-2.5">
        {factors.map((factor) => {
          const pct = Math.round((Math.abs(factor.shap_value ?? 0) / maxAbs) * 100);
          const pushesTowardAlert = (factor.shap_value ?? 0) > 0;
          const source = SOURCE_META[factor.source] ?? SOURCE_META.xgboost;

          return (
            <li key={factor.feature}>
              <div className="mb-1 flex items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <span className={`shrink-0 rounded border px-1 py-0.5 font-mono text-[10px] font-semibold ${source.className}`}>
                    {source.label}
                  </span>
                  <span className="truncate text-xs text-ink-dim">{factor.description}</span>
                </div>
                <span className="shrink-0 font-mono text-2xs text-ink-faint">
                  {factor.raw_value != null ? factor.raw_value.toFixed(2) : '—'}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-raised">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: pushesTowardAlert ? 'var(--sev-high)' : 'var(--accent-signal)',
                  }}
                />
              </div>
            </li>
          );
        })}
        {factors.length === 0 && <li className="text-xs text-ink-faint">No contributing factors recorded for this alert.</li>}
      </ul>
    </div>
  );
}
