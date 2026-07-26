# Sentinel — SOC Dashboard Frontend

React + Tailwind + Recharts dashboard for the behavioral anomaly
detection stack, wired to the Step 3 FastAPI backend (`/api/*` REST +
`/ws/stream` WebSocket).

## Run it

```bash
npm install
cp .env.example .env   # point VITE_API_BASE_URL at your FastAPI backend
npm run dev
```

Requires the Step 3 backend running (see its README) — the dashboard
seeds from `GET /api/alerts` + `GET /api/metrics` on load, then streams
live over `/ws/stream`.

## Structure

```
src/
  api/client.js                 REST client + WS URL resolver
  hooks/useWebSocketStream.js   Generic reconnecting WebSocket hook
  context/AnomalyStreamProvider.jsx  Single shared WS connection + derived state
  constants.js                  Severity/category labels, colors, formatters
  components/
    layout/     AppShell, Sidebar, HeaderStatusBar, LivePulseStrip
    charts/     RequestVolumeChart, AnomalyByCategoryChart, RiskScoreDistributionChart
    alerts/     AlertQueue, SeverityBadge / CategoryBadge
    entity/     EntityHistoryPanel (drill-down slide-over)
    explainability/  ContributingFactors (SHAP breakdown)
  pages/Dashboard.jsx            Composes everything above
  App.jsx                        Provider + shell + page
```

## Design system

Dark "ops telemetry" palette — near-black slate panels, a teal `signal`
accent for live/normal state, and a functional 4-step severity scale
(`sev-low/medium/high/critical`) that's reused everywhere risk shows up
(badges, charts, the entity baseline comparison). Data-dense values
(entity IDs, scores, timestamps) render in IBM Plex Mono; UI chrome and
prose render in Inter. All tokens live as CSS variables in
`src/styles/index.css` and are exposed as Tailwind classes via
`tailwind.config.js` (`bg-void`, `text-signal`, `text-sev-critical`, …) —
retune the palette in one place.

The signature element is the **live pulse strip** under the header: a
literal EKG-style ticker of the raw event stream (teal for normal,
severity-colored for anomalies), so the dashboard always shows the raw
feed underneath its summaries, not just aggregates.

## Wiring notes

- **One WebSocket connection.** `AnomalyStreamProvider` owns it;
  `AlertQueue`, all three charts, and the header status pill all read
  derived state from `useAnomalyStream()` rather than opening their own
  sockets or polling.
- **Live updates, no refresh.** A `{"type": "alert", ...}` frame
  prepends to the alert queue, increments the category bar chart's
  count for that `alert_type`, and (via the shared `events` buffer)
  feeds both the pulse strip and the request-volume chart. A
  `{"type": "event", ...}` frame updates the same buffer for normal
  traffic.
- **Reconnect behavior.** `useWebSocketStream` retries with capped
  exponential backoff (750ms → 15s) and the header reflects
  `connecting` / `open` / `closed` / `error` states live.
- **`EntityHistoryPanel`** fetches fresh on each "Details" click rather
  than reusing cached queue data, so the baseline comparison reflects
  the live River model state at click time, not at alert time.

## Extending

- `Sidebar`'s "Alerts" / "Entities" / "Audit Log" nav items are wired to
  `onNavigate` but `App.jsx` only renders `Dashboard` for `overview` —
  add routing (react-router, etc.) and additional pages as the app
  grows; every component requested for this deliverable already lives
  on the Overview screen.
- `AlertQueue` rows call `onSelectAlert`; wire a bulk-action toolbar or
  right-click menu there if you want ack/resolve from the table itself
  (the REST client already exposes `api.updateAlertStatus`, used by
  `AnomalyStreamProvider.acknowledgeAlert`).
