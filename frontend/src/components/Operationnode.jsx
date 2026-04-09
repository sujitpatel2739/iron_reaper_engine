import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { useGraphStore } from '../store/Usegraphstore';

const OP_SYMBOLS = {
  AddNode: '+', SubNode: '−', MulNode: '×', DivNode: '÷',
  SqNode: 'x²', NegNode: '−x', SqrtNode: '√', ScaleNode: '·α',
  ClipNode: '⌐¬', ConcatNode: '⊕', SplitNode: '⊘',
};

const STATUS_COLORS = {
  idle:    'var(--border-bright)',
  pending: 'var(--amber)',
  running: 'var(--phosphor)',
  done:    'var(--blue-dim)',
  error:   'var(--red)',
  locked:  'var(--text-muted)',
  branch:  'var(--blue)',
};

// Show #node_N by extracting trailing number
function nodeLabel(id) {
  const n = String(id).match(/\d+$/);
  return n ? `#node_${n[0]}` : `#${id}`;
}

const OperationNode = memo(({ id, data, selected }) => {
  const { updateNodeConfig, setSelectedNode, runMode, nodeTypes } = useGraphStore();
  const locked   = runMode !== 'idle';
  const color    = STATUS_COLORS[data.status] ?? STATUS_COLORS.idle;
  const symbol   = OP_SYMBOLS[data.nodeType] ?? '?';
  const typeDef  = nodeTypes[data.nodeType] ?? { fields: [] };

  const inputCount   = typeDef.inputs;
  const isMultiInput = inputCount === 'N' || inputCount > 1;

  return (
    <div
      onClick={() => setSelectedNode(id)}
      style={{
        width: 72,
        height: 72,
        borderRadius: '50%',
        background: selected ? 'var(--bg-elevated)' : 'var(--bg-surface)',
        border: `1.5px solid ${color}`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: locked ? 'default' : 'pointer',
        opacity: data.status === 'locked' ? 0.45 : 1,
        boxShadow: selected ? `0 0 16px ${color}33` : 'none',
        transition: 'all 0.15s',
        position: 'relative',
      }}
    >
      {/* Symbol */}
      <span style={{ fontSize: 18, fontFamily: 'var(--font-mono)', color, lineHeight: 1 }}>
        {symbol}
      </span>

      {/* Type label */}
      <span style={{
        fontSize: 7, fontFamily: 'var(--font-display)',
        color: 'var(--text-muted)', letterSpacing: '0.08em', marginTop: 2,
      }}>
        {data.nodeType?.replace('Node', '')}
      </span>

      {/* Node id — below circle */}
      <span style={{
        position: 'absolute',
        bottom: -16,
        fontSize: 9,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
        whiteSpace: 'nowrap',
      }}>
        {nodeLabel(id)}
      </span>

      {/* Config fields — below id label */}
      {typeDef.fields.length > 0 && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: '50%',
          transform: 'translateX(-50%)',
          marginTop: 22,
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          padding: '4px 8px',
          whiteSpace: 'nowrap',
          zIndex: 10,
        }}>
          {typeDef.fields.map(f => (
            <div key={f.name} style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 2 }}>
              <span style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {f.name}
              </span>
              <input
                type="number"
                step={f.type === 'float' ? '0.1' : '1'}
                value={data.config?.[f.name] ?? ''}
                disabled={locked}
                onChange={e => updateNodeConfig(
                  id, f.name,
                  f.type === 'float' ? parseFloat(e.target.value) : parseInt(e.target.value)
                )}
                style={{ width: 52, fontSize: 10, padding: '1px 4px' }}
              />
            </div>
          ))}
        </div>
      )}

      {/* Left = inputs, Right = output */}
      {isMultiInput ? (
        <>
          <Handle type="target" position={Position.Left} id="in_0"
            style={{ left: -5, top: '35%', transform: 'translateY(-50%)' }} />
          <Handle type="target" position={Position.Left} id="in_1"
            style={{ left: -5, top: '65%', transform: 'translateY(-50%)' }} />
        </>
      ) : (
        <Handle type="target" position={Position.Left} id="in_0"
          style={{ left: -5, top: '50%', transform: 'translateY(-50%)' }} />
      )}
      <Handle type="source" position={Position.Right} id="out_0"
        style={{ right: -5, top: '50%', transform: 'translateY(-50%)' }} />

      {/* Warnings dot */}
      {data.warnings?.length > 0 && (
        <div style={{
          position: 'absolute', top: -4, right: -4,
          width: 10, height: 10, borderRadius: '50%',
          background: 'var(--amber)',
          border: '1.5px solid var(--bg-void)',
        }} />
      )}
    </div>
  );
});

export default OperationNode;