/**
 * components/charts/AnomalyByCategoryChart.jsx
 * ================================================
 * Horizontal bar breakdown of alert volume by attack category, sourced
 * from `metrics.alerts_by_type` (already aggregated server-side by
 * `GET /api/metrics`) and kept live via AnomalyStreamProvider's
 * incremental updates on each new WebSocket alert.
 */

import { useMemo } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';
import { ChartCard } from './ChartCard';
import { categoryLabel } from '../../constants';

// Ranked by typical severity so the chart reads consistently across
// refreshes rather than reshuffling by count alone.
const CATEGORY_ORDER = [
  'credential_stuffing',
  'brute_force',
  'lateral_movement',
  'impossible_travel',
  'low_and_slow_exfiltration',
  'device_spoofing',
  'insider_drift',
];

export function AnomalyByCategoryChart({ alertsByType }) {
  const data = useMemo(() => {
    const entries = CATEGORY_ORDER.map((key) => ({
      key,
      label: categoryLabel(key, 'short'),
      count: alertsByType?.[key] ?? 0,
    }));
    // include any category not in our known order (forward-compatible)
    Object.keys(alertsByType ?? {})
      .filter((key) => !CATEGORY_ORDER.includes(key))
      .forEach((key) => entries.push({ key, label: categoryLabel(key, 'short'), count: alertsByType[key] }));
    return entries.sort((a, b) => b.count - a.count);
  }, [alertsByType]);

  return (
    <ChartCard title="Anomaly Rate by Category" subtitle="Total flagged alerts, by attack pattern">
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--border-hairline)" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            allowDecimals={false}
            tick={{ fill: 'var(--text-ink-faint)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            axisLine={{ stroke: 'var(--border-hairline)' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: 'var(--text-ink-dim)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={92}
          />
          <Tooltip
            cursor={{ fill: 'var(--bg-panel-raised)' }}
            contentStyle={{
              background: 'var(--bg-panel-raised)',
              border: '1px solid var(--border-hairline)',
              borderRadius: 6,
              fontSize: 12,
              fontFamily: 'IBM Plex Mono',
            }}
            labelStyle={{ color: 'var(--text-ink)' }}
          />
          <Bar dataKey="count" radius={[0, 3, 3, 0]} maxBarSize={16} isAnimationActive={false}>
            {data.map((entry, i) => (
              <Cell key={entry.key} fill={i === 0 ? 'var(--sev-critical)' : 'var(--sev-high)'} fillOpacity={i === 0 ? 1 : 0.75} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
