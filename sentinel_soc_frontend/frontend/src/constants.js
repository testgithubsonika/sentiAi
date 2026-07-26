/**
 * constants.js
 * ============
 * Single source of truth for severity/category labels + colors, so the
 * badge in the alert queue, the bar chart, and the entity panel all agree
 * with each other (and with `models.AlertSeverity` / `models.LabelType`
 * on the backend).
 */

export const SEVERITY = {
  low: { label: 'Low', color: 'var(--sev-low)', className: 'text-sev-low border-sev-low/30 bg-sev-low/10' },
  medium: { label: 'Medium', color: 'var(--sev-medium)', className: 'text-sev-medium border-sev-medium/30 bg-sev-medium/10' },
  high: { label: 'High', color: 'var(--sev-high)', className: 'text-sev-high border-sev-high/30 bg-sev-high/10' },
  critical: { label: 'Critical', color: 'var(--sev-critical)', className: 'text-sev-critical border-sev-critical/30 bg-sev-critical/10' },
};

export const severityFromRisk = (score) => {
  if (score >= 0.85) return 'critical';
  if (score >= 0.6) return 'high';
  if (score >= 0.35) return 'medium';
  return 'low';
};

// Mirrors models.LabelType (minus "normal"). Short labels for badges,
// full labels for chart axes/legends.
export const ATTACK_CATEGORIES = {
  brute_force: { short: 'Brute Force', full: 'Brute Force' },
  impossible_travel: { short: 'Imp. Travel', full: 'Impossible Travel' },
  credential_stuffing: { short: 'Cred Stuffing', full: 'Credential Stuffing' },
  lateral_movement: { short: 'Lateral Move', full: 'Lateral Movement' },
  device_spoofing: { short: 'Device Spoof', full: 'Device Spoofing' },
  low_and_slow_exfiltration: { short: 'Slow Exfil', full: 'Low & Slow Exfiltration' },
  insider_drift: { short: 'Insider Drift', full: 'Insider Drift' },
};

export const categoryLabel = (key, variant = 'short') =>
  ATTACK_CATEGORIES[key]?.[variant] ?? key?.replace(/_/g, ' ') ?? 'Unknown';

export const formatScore = (score) => (score == null ? '—' : score.toFixed(2));

export const formatTime = (iso) => {
  const d = typeof iso === 'string' ? new Date(iso) : iso;
  if (!d || Number.isNaN(d.getTime())) return '—';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

export const formatRelative = (iso) => {
  const d = typeof iso === 'string' ? new Date(iso) : iso;
  if (!d || Number.isNaN(d.getTime())) return '—';
  const seconds = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
};
