import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { useGraphStore } from '../store/Usegraphstore';

const STATUS_COLORS = {
  idle:    'var(--border-bright)',
  pending: 'var(--amber)',
  running: 'var(--phosphor)',
  done:    'var(--phosphor-dim)',
  error:   'var(--red)',
  locked:  'var(--text-muted)',
  branch:  'var(--blue)',
};

const STATUS_GLOW = {
  running: '0 0 16px rgba(0,255,136,0.4)',
  pending: '0 0 12px rgba(255,170,0,0.3)',
  error:   '0 0 12px rgba(255,51,85,0.3)',
  branch:  '0 0 12px rgba(0,153,255,0.3)',
  done:    'none',
  idle:    'none',
  locked:  'none',
};

function FieldInput({ field, value, onChange, locked }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 10, minWidth: 72, fontFamily: 'var(--font-mono)' }}>
        {field.name}
      </span>
      <input
        type="number"
        step={field.type === 'float' ? '0.00001' : '1'}
        value={value ?? ''}
        disabled={locked}
        onChange={e => {
          const v = field.type === 'float'
            ? parseFloat(e.target.value)
            : parseInt(e.target.value, 10);
          onChange(v);
        }}
        style={{ width: 70, fontSize: 11, padding: '2px 6px' }}
      />
    </div>
  );
}

// Show #layer_N instead of #node_N by extracting the trailing number
function layerLabel(id) {
  const n = String(id).match(/\d+$/);
  return n ? `#layer_${n[0]}` : `#${id}`;
}

const LayerNode = memo(({ id, data, selected }) => {
  const { updateNodeConfig, setSelectedNode, runMode, layerTypes } = useGraphStore();
  const locked  = runMode !== 'idle';
  const color   = STATUS_COLORS[data.status] ?? STATUS_COLORS.idle;
  const glow    = STATUS_GLOW[data.status]   ?? 'none';
  const typeDef = layerTypes[data.layerType] ?? { fields: [] };

  return (
    <div
      onClick={() => setSelectedNode(id)}
      style={{
        minWidth: 160,
        background: selected ? 'var(--bg-elevated)' : 'var(--bg-surface)',
        border: `1.5px solid ${selected ? color : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        boxShadow: selected ? glow : 'none',
        transition: 'all 0.15s',
        cursor: locked ? 'default' : 'pointer',
        opacity: data.status === 'locked' ? 0.45 : 1,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Status bar — top edge */}
      <div style={{
        height: 2,
        background: color,
        opacity: data.status === 'idle' ? 0.3 : 1,
        boxShadow: data.status === 'running' ? '0 0 8px var(--phosphor)' : 'none',
        transition: 'background 0.2s',
      }} />

      {/* Header */}
      <div style={{
        padding: '6px 10px 4px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 9,
          letterSpacing: '0.12em',
          color,
          textTransform: 'uppercase',
        }}>
          {data.layerType}
        </span>
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          {layerLabel(id)}
        </span>
      </div>

      {/* Config fields */}
      <div style={{ padding: '6px 10px 8px' }}>
        {typeDef.fields.length === 0 && (
          <span style={{ color: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
            no config
          </span>
        )}
        {typeDef.fields.map(f => (
          <FieldInput
            key={f.name}
            field={f}
            value={data.config?.[f.name]}
            onChange={v => updateNodeConfig(id, f.name, v)}
            locked={locked}
          />
        ))}

        {data.metrics?.input_shape && (
          <div style={{ marginTop: 6, display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              {data.metrics.input_shape.join('×')}
            </span>
            <span style={{ fontSize: 9, color: 'var(--border-bright)' }}>→</span>
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--phosphor-dim)' }}>
              {data.metrics.output_shape?.join('×')}
            </span>
          </div>
        )}

        {data.warnings?.length > 0 && (
          <div style={{ marginTop: 4 }}>
            {data.warnings.map((w, i) => (
              <div key={i} style={{
                fontSize: 9, color: 'var(--amber)', fontFamily: 'var(--font-mono)',
                display: 'flex', gap: 4, alignItems: 'flex-start',
              }}>
                <span>⚠</span><span>{w}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Left = input, Right = output — horizontal flow */}
      <Handle
        type="target"
        position={Position.Left}
        style={{ left: -5, top: '50%', transform: 'translateY(-50%)' }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ right: -5, top: '50%', transform: 'translateY(-50%)' }}
      />
    </div>
  );
});

export default LayerNode;