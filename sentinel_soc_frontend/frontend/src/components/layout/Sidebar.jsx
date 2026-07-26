/**
 * components/layout/Sidebar.jsx
 */

import { LayoutGrid, ShieldAlert, Users, FileClock, Radio } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: LayoutGrid },
  { id: 'alerts', label: 'Alert Queue', icon: ShieldAlert },
  { id: 'entities', label: 'Entities', icon: Users },
  { id: 'audit', label: 'Audit Log', icon: FileClock },
];

export function Sidebar({ activeId = 'overview', onNavigate }) {
  return (
    <aside className="hidden w-56 shrink-0 flex-col border-r border-hairline bg-panel md:flex">
      <div className="flex h-14 items-center gap-2 border-b border-hairline px-5">
        <Radio className="h-4 w-4 text-signal" strokeWidth={2.5} />
        <span className="font-mono text-sm font-semibold tracking-tight text-ink">SENTINEL</span>
      </div>

      <nav className="flex-1 space-y-0.5 p-3">
        <p className="px-2 pb-2 pt-1 font-mono text-2xs uppercase tracking-widest text-ink-faint">
          Monitoring
        </p>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const active = id === activeId;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onNavigate?.(id)}
              className={`flex w-full items-center gap-3 rounded px-3 py-2 text-sm transition-colors ${
                active
                  ? 'bg-panel-raised text-ink'
                  : 'text-ink-dim hover:bg-panel-raised/60 hover:text-ink'
              }`}
            >
              <Icon
                className={`h-4 w-4 ${active ? 'text-signal' : 'text-ink-faint'}`}
                strokeWidth={2}
              />
              {label}
              {active && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-signal" />}
            </button>
          );
        })}
      </nav>

      <div className="border-t border-hairline p-4">
        <p className="font-mono text-2xs leading-relaxed text-ink-faint">
          Behavioral Anomaly Detection
          <br />
          IForest &middot; Bi-LSTM &middot; XGBoost
        </p>
      </div>
    </aside>
  );
}
