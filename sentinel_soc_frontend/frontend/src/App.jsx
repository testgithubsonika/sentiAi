/**
 * App.jsx
 * ========
 * Root component. Wraps the dashboard in AnomalyStreamProvider (single
 * WebSocket connection + REST seed, shared by every chart/list) and the
 * AppShell layout (sidebar, header, live pulse strip).
 */

import { useState } from 'react';
import { AnomalyStreamProvider } from './context/AnomalyStreamProvider';
import { AppShell } from './components/layout/AppShell';
import { Dashboard } from './pages/Dashboard';

export default function App() {
  const [activeNavId, setActiveNavId] = useState('overview');

  return (
    <AnomalyStreamProvider>
      <AppShell activeNavId={activeNavId} onNavigate={setActiveNavId}>
        {/* Alerts / Entities / Audit nav destinations are left as an
            exercise for routing (react-router etc.) -- Overview covers
            every component requested in this deliverable. */}
        <Dashboard />
      </AppShell>
    </AnomalyStreamProvider>
  );
}
