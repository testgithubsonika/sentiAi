/**
 * components/charts/RequestVolumeChart.jsx
 * ===========================================
 * Line chart of normal vs. anomalous event volume. Buckets the live
 * event buffer client-side into fixed-width time windows (default 30s
 * buckets over a 10 minute rolling window) and re-derives on every new
 * event, so the chart animates forward in real time as the WebSocket
 * stream feeds it -- no polling.
 */

import { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { ChartCard } from './ChartCard';

const BUCKET_SECONDS = 30;
const BUCKET_COUNT = 20;

function bucketize(events) {
  const now = Date.now();
  const bucketMs = BUCKET_SECONDS * 1000;
  const buckets = Array.from({ length: BUCKET_COUNT }, (_, i) => {
    const bucketStart = now - (BUCKET_COUNT - i) * bucketMs;
    return {
      time: new Date(bucketStart),
      normal: 0,
      anomalous: 0,
    };
  });

  for (const event of events) {
    const ts = event.ts instanceof Date ? event.ts.getTime() : new Date(event.ts).getTime();
    const age = now - ts;
    if (age < 0 || age > BUCKET_COUNT * bucketMs) continue;
    const index = BUCKET_COUNT - 1 - Math.floor(age / bucketMs);
    if (index < 0 || index >= BUCKET_COUNT) continue;
    if (event.kind === 'anomalous') buckets[index].anomalous += 1;
    else buckets[index].normal += 1;
  }

  return buckets.map((b) => ({
    label: b.time.toLocaleTimeString([], { minute: '2-digit', second: '2-digit' }),
    normal: b.normal,
    anomalous: b.anomalous,
  }));
}

export function RequestVolumeChart({ events }) {
  const data = useMemo(() => bucketize(events), [events]);

  return (
    <ChartCard
      title="Request Volume"
      subtitle={`Normal vs. anomalous traffic — last ${(BUCKET_SECONDS * BUCKET_COUNT) / 60} min`}
    >
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke="var(--border-hairline)" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: 'var(--text-ink-faint)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            axisLine={{ stroke: 'var(--border-hairline)' }}
            tickLine={false}
            interval={4}
          />
          <YAxis
            tick={{ fill: 'var(--text-ink-faint)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            axisLine={false}
            tickLine={false}
            width={32}
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
          />
          <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'Inter' }} iconType="circle" iconSize={8} />
          <Line
            type="monotone"
            dataKey="normal"
            name="Normal"
            stroke="var(--accent-signal)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="anomalous"
            name="Anomalous"
            stroke="var(--sev-high)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
