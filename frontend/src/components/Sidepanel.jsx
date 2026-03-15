import React, { useState } from 'react';
import MetricsPanel from './Metricspanel';
import ObserverConfig from './Observerconfig';
import StepControls from './Stepcontrols';
import { useGraphStore } from '../store/Usegraphstore';

const TABS = ['METRICS', 'CONFIG', 'STEP'];

export default function SidePanel() {
  const [tab, setTab] = useState('METRICS');
  const { selectedNodeId, nodes, runMode } = useGraphStore();
  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  const isStepping = runMode === 'stepping';

  return (
    <div style={{
      width: 280,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-panel)',
      borderLeft: '1px solid var(--border)',
      flexShrink: 0,
    }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: '8px 0',
              fontFamily: 'var(--font-display)',
              fontSize: 8,
              letterSpacing: '0.12em',
              color: tab === t ? 'var(--phosphor)' : 'var(--text-muted)',
              borderBottom: tab === t ? '2px solid var(--phosphor)' : '2px solid transparent',
              marginBottom: -1,
              transition: 'all 0.15s',
              background: 'none',
              cursor: 'pointer',
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Panel body */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 'METRICS' && <MetricsPanel node={selectedNode} />}
        {tab === 'CONFIG'  && <ObserverConfig />}
        {tab === 'STEP'    && <StepControls />}
      </div>
    </div>
  );
}