import React, { useRef, useState } from 'react';
import { useGraphStore } from '../store/Usegraphstore';
import { validateNetwork, importModel, saveNetwork, loadNetwork, runFull } from '../api/api';

function DraggablePill({ label, nodeType, kind, color = 'var(--text-secondary)' }) {
  return (
    <div
      draggable
      onDragStart={e => {
        e.dataTransfer.setData('nodeType', nodeType);
        e.dataTransfer.setData('nodeKind', kind);
      }}
      style={{
        padding: '3px 10px',
        borderRadius: 2,
        border: `1px solid var(--border)`,
        background: 'var(--bg-surface)',
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color,
        cursor: 'grab',
        userSelect: 'none',
        whiteSpace: 'nowrap',
        transition: 'all 0.12s',
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = color; e.currentTarget.style.color = color; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = color; }}
    >
      {label}
    </div>
  );
}

function Divider() {
  return <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px', flexShrink: 0 }} />;
}

function ToolBtn({ children, onClick, color = 'var(--text-secondary)', title, disabled }) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      style={{
        padding: '4px 10px',
        borderRadius: 3,
        border: '1px solid var(--border)',
        background: 'var(--bg-surface)',
        color: disabled ? 'var(--text-muted)' : color,
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        cursor: disabled ? 'not-allowed' : 'pointer',
        whiteSpace: 'nowrap',
        transition: 'all 0.12s',
        opacity: disabled ? 0.5 : 1,
      }}
      onMouseEnter={e => { if (!disabled) e.currentTarget.style.borderColor = color; }}
      onMouseLeave={e => { if (!disabled) e.currentTarget.style.borderColor = 'var(--border)'; }}
    >
      {children}
    </button>
  );
}

export default function Toolbar() {
  const importRef = useRef();
  const loadRef   = useRef();

  const {
    runMode, setRunMode,
    nodes, edges,
    runConfig, inputFile,
    layerTypes, nodeTypes,
    setNodeWarnings, setAllNodeStatuses, setNodeMetrics,
    loadGraph, clearGraph,
    layoutDirection, toggleLayout,
  } = useGraphStore();

  const locked = runMode !== 'idle';

  const handleValidate = async () => {
    const result = await validateNetwork({ nodes, edges });
    (result.warnings ?? []).forEach(w => setNodeWarnings(w.node_id, [w.message]));
    if (result.warnings?.length === 0) alert('Network is valid ✓');
  };

  const handleRun = async () => {
    if (!nodes.length) return;
    setRunMode('running');
    setAllNodeStatuses('locked');
    try {
      const report = await runFull({ nodes, edges }, runConfig, inputFile);
      // Distribute metrics to nodes
      (report.raw_metrics ?? []).forEach(lm => {
        setNodeMetrics(lm.layer_name, lm.metrics);
      });
      setAllNodeStatuses('done');
      setRunMode('done');
    } catch {
      setRunMode('error');
    }
  };

  const handleImport = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const graph = await importModel(file);
      if (graph?.detail) {
        alert(`Import failed:\n${graph.detail}`);
      } else if (!graph?.nodes) {
        alert('Import failed: server returned an unexpected response.');
      } else {
        loadGraph(graph);
      }
    } catch (err) {
      alert(`Import failed: ${err.message}`);
    }
    e.target.value = '';
  };

  const handleLoad = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const graph = await loadNetwork(file);
      loadGraph(graph);
    } catch { alert('Invalid network file.'); }
    e.target.value = '';
  };

  return (
    <div style={{
      height: 44,
      display: 'flex',
      alignItems: 'center',
      gap: 6,
      padding: '0 14px',
      background: 'var(--bg-panel)',
      borderBottom: '1px solid var(--border)',
      flexShrink: 0,
      overflowX: 'auto',
    }}>
      {/* Logo */}
      <span style={{
        fontFamily: 'var(--font-display)',
        fontSize: 11,
        color: 'var(--phosphor)',
        letterSpacing: '0.2em',
        whiteSpace: 'nowrap',
        marginRight: 8,
        flexShrink: 0,
      }}>
        NSO
      </span>

      <Divider />

      {/* Layer palette */}
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', flexShrink: 0 }}>LAYERS</span>
      {Object.keys(layerTypes).map(t => (
        <DraggablePill key={t} label={t} nodeType={t} kind="layer" color="var(--phosphor-dim)" />
      ))}

      <Divider />

      {/* Node palette */}
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', flexShrink: 0 }}>NODES</span>
      {Object.keys(nodeTypes).map(t => (
        <DraggablePill key={t} label={t.replace('Node','')} nodeType={t} kind="operation" color="var(--blue-dim)" />
      ))}

      <Divider />

      {/* Actions */}
      <ToolBtn onClick={handleValidate} disabled={locked} color="var(--amber)" title="Validate shapes">
        ⚡ Validate
      </ToolBtn>
      <ToolBtn
        onClick={handleRun}
        disabled={locked || !nodes.length}
        color="var(--phosphor)"
        title="Run full diagnostic"
      >
        ▶ Run
      </ToolBtn>
      {runMode !== 'idle' && (
        <ToolBtn
          onClick={() => { setRunMode('idle'); setAllNodeStatuses('idle'); }}
          color="var(--red)"
        >
          ■ Stop
        </ToolBtn>
      )}

      <Divider />

      {/* Layout toggle */}
      <ToolBtn onClick={toggleLayout} title="Toggle layout direction">
        {layoutDirection === 'TB' ? '↔ Horizontal' : '↕ Vertical'}
      </ToolBtn>

      <Divider />

      {/* File operations */}
      <ToolBtn onClick={() => !locked && saveNetwork({ nodes, edges, run_config: runConfig })} disabled={locked}>
        ↓ Save
      </ToolBtn>
      <ToolBtn onClick={() => !locked && loadRef.current.click()} disabled={locked}>
        ↑ Load
      </ToolBtn>
      <ToolBtn onClick={() => !locked && importRef.current.click()} disabled={locked} color="var(--amber)">
        ⬆ Import Model
      </ToolBtn>
      <ToolBtn onClick={() => !locked && clearGraph()} disabled={locked} color="var(--red)">
        ✕ Clear
      </ToolBtn>

      <input ref={importRef} type="file" accept=".pt,.pth,.h5" style={{ display: 'none' }} onChange={handleImport} />
      <input ref={loadRef}   type="file" accept=".json"        style={{ display: 'none' }} onChange={handleLoad} />
    </div>
  );
}