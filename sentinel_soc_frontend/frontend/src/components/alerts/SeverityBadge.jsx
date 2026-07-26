/**
 * components/alerts/SeverityBadge.jsx
 */

import { SEVERITY, categoryLabel } from '../../constants';

export function SeverityBadge({ severity }) {
  const meta = SEVERITY[severity] ?? SEVERITY.low;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-2xs font-semibold uppercase tracking-wide ${meta.className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: meta.color }} />
      {meta.label}
    </span>
  );
}

export function CategoryBadge({ category }) {
  return (
    <span className="inline-flex items-center rounded border border-hairline bg-panel-raised px-1.5 py-0.5 font-mono text-2xs text-ink-dim">
      {categoryLabel(category, 'short')}
    </span>
  );
}
