/**
 * api/client.js
 * =============
 * Thin fetch wrapper around the FastAPI backend from Step 3
 * (`/api/entities`, `/api/alerts`, `/api/metrics`). Reads the base URL
 * from Vite's `import.meta.env.VITE_API_BASE_URL`, falling back to
 * same-origin `/api` for a reverse-proxied deployment.
 */

const BASE_URL = import.meta.env?.VITE_API_BASE_URL ?? '';

async function request(path, { params, ...init } = {}) {
  const url = new URL(`${BASE_URL}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        url.searchParams.set(key, value);
      }
    });
  }

  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${init.method ?? 'GET'} ${path} failed (${res.status}): ${body}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  listEntities: (params) => request('/api/entities', { params }),
  getEntityHistory: (entityId, params) => request(`/api/entities/${encodeURIComponent(entityId)}/history`, { params }),
  listAlerts: (params) => request('/api/alerts', { params }),
  updateAlertStatus: (alertId, body) =>
    request(`/api/alerts/${alertId}`, { method: 'PATCH', body: JSON.stringify(body) }),
  getMetrics: () => request('/api/metrics'),
};

/** Derives the ws:// or wss:// stream URL from the same base as the REST API. */
export function getStreamUrl() {
  if (BASE_URL) {
    return BASE_URL.replace(/^http/, 'ws') + '/ws/stream';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}/ws/stream`;
}
