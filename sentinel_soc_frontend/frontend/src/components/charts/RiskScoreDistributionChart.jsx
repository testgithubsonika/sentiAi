/**
 * components/charts/RiskScoreDistributionChart.jsx
 * ====================================================
 * Histogram of alert risk scores (0-1, 10 bins), rendered as an area
 * chart with a gradient that shifts from the signal teal at low risk to
 * critical rose at high risk -- the fill color itself encodes the same
 * severity scale as the badges, so the shape of the distribution reads
 * as a risk profile, not just a count.
 */

import { useMemo } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { ChartCard } from './ChartCard';

const BIN_COUNT = 10;

function buildHistogram(alerts) {
  const bins = Array.from({ length: BIN_COUNT }, (_, i) => ({
    bin: `${(i / BIN_COUNT).toFixed(1)}–${((i + 1) / BIN_COUNT).toFixed(1)}`,
    count: 0,
  }));
  for (const alert of alerts) {
    const score = Math.min(0.999, Math.max(0, alert.risk_score ?? 0));
    const index = Math.floor(score * BIN_COUNT);
    bins[index].count += 1;
  }
  return bins;
}

export function RiskScoreDistributionChart({ alerts }) {
  const data = useMemo(() => buildHistogram(alerts), [alerts]);

  return (
    <ChartCard title="Risk Score Distribution" subtitle={`${alerts.length} scored alerts in queue`}>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={data} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
          <defs>
            <linearGradient id="riskGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="var(--accent-signal)" />
              <stop offset="45%" stopColor="var(--sev-medium)" />
              <stop offset="75%" stopColor="var(--sev-high)" />
              <stop offset="100%" stopColor="var(--sev-critical)" />
            </linearGradient>
            <linearGradient id="riskFade" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--sev-high)" stopOpacity={0.35} />
              <stop offset="100%" stopColor="var(--sev-high)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="var(--border-hairline)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="bin"
            tick={{ fill: 'var(--text-ink-faint)', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
            axisLine={{ stroke: 'var(--border-hairline)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'var(--text-ink-faint)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            axisLine={false}
            tickLine={false}
            width={28}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-panel-raised)',
              border: '1px solid var(--border-hairline)',
              borderRadius: 6,
              fontSize: 12,
              fontFamily: 'IBM Plex Mono',
            }}
            labelStyle={{ color: 'var(--text-ink-dim)' }}
            formatter={(value) => [value, 'Alerts']}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="url(#riskGradient)"
            strokeWidth={2}
            fill="url(#riskFade)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
