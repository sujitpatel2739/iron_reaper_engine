import React, { useEffect } from 'react';
import { ReactFlowProvider } from 'reactflow';
import Toolbar from './components/Toolbar';
import GraphCanvas from './components/Graphcanvas';
import SidePanel from './components/Sidepanel';
import { useGraphStore } from './store/Usegraphstore';
import { fetchLayerTypes, fetchNodeTypes } from './api/api';
import './styles/global.css';

export default function App() {
  const { setLayerTypes, setNodeTypes } = useGraphStore();

  // Hydrate type registries from backend on mount
  useEffect(() => {
    fetchLayerTypes()
      .then(data => setLayerTypes(data))
      .catch(() => { /* use defaults */ });
    fetchNodeTypes()
      .then(data => setNodeTypes(data))
      .catch(() => { /* use defaults */ });
  }, []);

  return (
    <ReactFlowProvider>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--bg-void)',
      }}>
        {/* Top toolbar */}
        <Toolbar />

        {/* Main area: canvas + side panel */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
          <GraphCanvas />
          <SidePanel />
        </div>
      </div>
    </ReactFlowProvider>
  );
}