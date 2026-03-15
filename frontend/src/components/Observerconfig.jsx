import React, { useRef } from 'react';
import { useGraphStore } from '../store/Usegraphstore';

const AVAILABLE_OBSERVERS = [
  { id: 'SignalStatsObserver',    label: 'Signal Stats',    desc: 'activation mean/var, grad norm/var' },
  { id: 'SignalShapeObserver',    label: 'Signal Shape',    desc: 'tensor shapes through the network' },
  { id: 'ResidualEnergyObserver', label: 'Residual Energy', desc: 'residual vs shortcut path energy' },
];

export default function ObserverConfig() {
  const { runConfig, setRunConfig, setInputFile, inputFile, runMode } = useGraphStore();
  const fileRef = useRef();
  const locked = runMode !== 'idle';

  const toggleObserver = (id) => {
    const current = runConfig.observers;
    const next = current.includes(id)
      ? current.filter(o => o !== id)
      : [...current, id];
    setRunConfig({ observers: next });
  };

  const parseShape = (str) => {
    try {
      return str.split(',').map(s => parseInt(s.trim(), 10)).filter(Boolean);
    } catch { return runConfig.input_shape; }
  };

  return (
    <div style={{ padding: 16, overflowY: 'auto', height: '100%' }}>

      {/* Observers */}
      <SectionLabel>Observers</SectionLabel>
      <div style={{ marginBottom: 20 }}>
        {AVAILABLE_OBSERVERS.map(obs => {
          const active = runConfig.observers.includes(obs.id);
          return (
            <div
              key={obs.id}
              onClick={() => !locked && toggleObserver(obs.id)}
              style={{
                padding: '8px 10px',
                marginBottom: 4,
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${active ? 'var(--phosphor-dim)' : 'var(--border)'}`,
                background: active ? 'var(--phosphor-faint)' : 'var(--bg-surface)',
                cursor: locked ? 'default' : 'pointer',
                opacity: locked ? 0.6 : 1,
                transition: 'all 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: 1,
                  background: active ? 'var(--phosphor)' : 'transparent',
                  border: `1.5px solid ${active ? 'var(--phosphor)' : 'var(--border-bright)'}`,
                  flexShrink: 0,
                  transition: 'all 0.15s',
                }} />
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: active ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {obs.label}
                </span>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 2, paddingLeft: 16 }}>
                {obs.desc}
              </div>
            </div>
          );
        })}
      </div>

      {/* Input shape */}
      <SectionLabel>Input Shape</SectionLabel>
      <input
        disabled={locked}
        defaultValue={runConfig.input_shape.join(', ')}
        onBlur={e => setRunConfig({ input_shape: parseShape(e.target.value) })}
        placeholder="32, 128"
        style={{ width: '100%', marginBottom: 16 }}
      />

      {/* Seed */}
      <SectionLabel>Random Seed</SectionLabel>
      <input
        type="number"
        disabled={locked}
        value={runConfig.seed ?? ''}
        onChange={e => setRunConfig({ seed: parseInt(e.target.value) || null })}
        placeholder="42"
        style={{ width: '100%', marginBottom: 20 }}
      />

      {/* Input data */}
      <SectionLabel>Input Data</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <button
          disabled={locked}
          onClick={() => fileRef.current.click()}
          style={{
            padding: '6px 12px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: inputFile ? 'var(--phosphor)' : 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            textAlign: 'left',
            cursor: locked ? 'default' : 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {inputFile ? `📄 ${inputFile.name}` : '+ Upload .npy / .csv'}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".npy,.csv,.pt"
          style={{ display: 'none' }}
          onChange={e => setInputFile(e.target.files[0] ?? null)}
        />
        {inputFile && (
          <button
            onClick={() => setInputFile(null)}
            style={{
              fontSize: 9, color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              textAlign: 'left',
              padding: '2px 0',
            }}
          >
            ✕ remove — use random input
          </button>
        )}
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
          If no file uploaded, a random tensor of the configured shape will be generated.
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)',
      fontSize: 9,
      color: 'var(--text-muted)',
      textTransform: 'uppercase',
      letterSpacing: '0.12em',
      marginBottom: 6,
    }}>
      {children}
    </div>
  );
}