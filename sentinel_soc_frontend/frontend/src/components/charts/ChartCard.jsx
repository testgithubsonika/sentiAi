/**
 * components/charts/ChartCard.jsx
 * A consistent panel frame for every chart: eyebrow-style title, muted
 * subtitle, hairline border, subtle inset shadow.
 */

export function ChartCard({ title, subtitle, action, children }) {
  return (
    <section className="rounded-md border border-hairline bg-panel p-4 shadow-panel">
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="font-mono text-2xs font-semibold uppercase tracking-widest text-ink-dim">
            {title}
          </h2>
          {subtitle && <p className="mt-0.5 text-xs text-ink-faint">{subtitle}</p>}
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}
