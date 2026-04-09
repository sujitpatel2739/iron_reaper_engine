import React, { memo, useState, useCallback } from 'react';
import { Handle, Position } from 'reactflow';
import { useGraphStore } from '../store/Usegraphstore';

const STATUS_COLORS = {
  idle:    'var(--border-bright)',
  pending: 'var(--amber)',
  running: 'var(--phosphor)',
  done:    'var(--phosphor-dim)',
  error:   'var(--red)',
  locked:  'var(--text-muted)',
};

const STATUS_GLOW = {
  running: '0 0 16px rgba(0,255,136,0.4)',
  pending: '0 0 12px rgba(255,170,0,0.3)',
  error:   '0 0 12px rgba(255,51,85,0.3)',
  done:    'none',
  idle:    'none',
  locked:  'none',
};

// ---------------------------------------------------------------------------
// Editable port label
// ---------------------------------------------------------------------------

function PortLabel({ label, portIndex, nodeId, locked }) {
  const { updateInputLabel } = useGraphStore();
  const [editing, setEditing] = useState(false);
  const [draft,   setDraft]   = useState(label);

  const commit = useCallback(() => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== label) updateInputLabel(nodeId, portIndex, trimmed);
    else setDraft(label);
    setEditing(false);
  }, [draft, label, nodeId, portIndex, updateInputLabel]);

  if (editing) {
    return (
      <input
        autoFocus
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => {
          if (e.key === 'Enter') commit();
          if (e.key === 'Escape') { setDraft(label); setEditing(false); }
        }}
        onClick={e => e.stopPropagation()}
        style={{
          width: 80, fontSize: 10, fontFamily: 'var(--font-mono)',
          padding: '1px 4px', background: 'var(--bg-elevated)',
          border: '1px solid var(--phosphor-dim)', borderRadius: 2,
          color: 'var(--text-primary)', outline: 'none',
        }}
      />
    );
  }

  return (
    <span
      onClick={e => { if (!locked) { e.stopPropagation(); setEditing(true); } }}
      title={locked ? label : 'Click to rename'}
      style={{
        fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--phosphor-dim)',
        cursor: locked ? 'default' : 'text',
        borderBottom: locked ? 'none' : '1px dashed var(--border-bright)',
        paddingBottom: 1, maxWidth: 90,
        overflow: 'hidden', textOverflow: 'ellipsis',
        whiteSpace: 'nowrap', display: 'inline-block',
      }}
    >
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Single port row
// ---------------------------------------------------------------------------

function InputPort({ label, portIndex, nodeId, locked, onRemove, canRemove }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '4px 10px', borderBottom: '1px solid var(--border)',
      position: 'relative', minHeight: 28, gap: 6,
    }}>
      <PortLabel label={label} portIndex={portIndex} nodeId={nodeId} locked={locked} />

      {/* Remove button */}
      {!locked && canRemove && (
        <button
          onClick={e => { e.stopPropagation(); onRemove(portIndex); }}
          title="Remove input"
          style={{
            width: 14, height: 14, borderRadius: 2,
            border: '1px solid var(--border)', background: 'transparent',
            color: 'var(--text-muted)', fontSize: 10, lineHeight: 1,
            cursor: 'pointer', display: 'flex', alignItems: 'center',
            justifyContent: 'center', flexShrink: 0, padding: 0,
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--red)'; e.currentTarget.style.color = 'var(--red)'; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
        >
          −
        </button>
      )}

      {/* Source handle per port — right side, carries this port's label as id */}
      <Handle
        type="source"
        position={Position.Right}
        id={`port-${portIndex}`}
        style={{ right: -6, top: '50%', transform: 'translateY(-50%)', width: 8, height: 8 }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// InputLayerNode
// ---------------------------------------------------------------------------

const InputLayerNode = memo(({ id, data, selected }) => {
  const {
    setSelectedNode, runMode,
    addInputPort, removeInputPort, updateNodeName,
  } = useGraphStore();

  const locked  = runMode !== 'idle';
  const glow    = STATUS_GLOW[data.status] ?? 'none';
  const inputs  = data.inputs ?? ['input_0'];

  const [editingName, setEditingName] = useState(false);
  const [nameDraft,   setNameDraft]   = useState(data.name ?? 'inputlayer');

  const commitName = useCallback(() => {
    const trimmed = nameDraft.trim();
    if (trimmed) updateNodeName(id, trimmed);
    else setNameDraft(data.name ?? 'inputlayer');
    setEditingName(false);
  }, [nameDraft, data.name, id, updateNodeName]);

  return (
    <div
      onClick={() => setSelectedNode(id)}
      style={{
        minWidth: 180,
        background: selected ? 'var(--bg-elevated)' : 'var(--bg-surface)',
        border: `1.5px solid ${selected ? 'var(--phosphor)' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        boxShadow: selected ? glow : 'none',
        transition: 'all 0.15s',
        cursor: locked ? 'default' : 'pointer',
        opacity: data.status === 'locked' ? 0.45 : 1,
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Status bar */}
      <div style={{
        height: 2, background: 'var(--phosphor)',
        opacity: data.status === 'idle' ? 0.4 : 1,
        borderRadius: '4px 4px 0 0', transition: 'opacity 0.2s',
      }} />

      {/* Header */}
      <div style={{
        padding: '6px 10px 5px', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', borderBottom: '1px solid var(--border)', gap: 6,
      }}>
        <span style={{
          fontFamily: 'var(--font-display)', fontSize: 8,
          letterSpacing: '0.14em', color: 'var(--phosphor)',
          textTransform: 'uppercase', flexShrink: 0,
        }}>
          INPUT
        </span>

        {editingName ? (
          <input
            autoFocus
            value={nameDraft}
            onChange={e => setNameDraft(e.target.value)}
            onBlur={commitName}
            onKeyDown={e => {
              if (e.key === 'Enter') commitName();
              if (e.key === 'Escape') { setNameDraft(data.name ?? 'inputlayer'); setEditingName(false); }
            }}
            onClick={e => e.stopPropagation()}
            style={{
              flex: 1, fontSize: 10, fontFamily: 'var(--font-mono)',
              padding: '1px 4px', background: 'var(--bg-elevated)',
              border: '1px solid var(--phosphor-dim)', borderRadius: 2,
              color: 'var(--text-primary)', outline: 'none', minWidth: 0,
            }}
          />
        ) : (
          <span
            onClick={e => { if (!locked) { e.stopPropagation(); setEditingName(true); } }}
            title={locked ? data.name : 'Click to rename layer'}
            style={{
              flex: 1, fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)', cursor: locked ? 'default' : 'text',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              borderBottom: locked ? 'none' : '1px dashed var(--border-bright)',
            }}
          >
            {data.name ?? 'inputlayer'}
          </span>
        )}

        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', flexShrink: 0 }}>
          #{id}
        </span>
      </div>

      {/* Input ports */}
      {inputs.map((label, i) => (
        <InputPort
          key={i}
          label={label}
          portIndex={i}
          nodeId={id}
          locked={locked}
          canRemove={inputs.length > 1}
          onRemove={(idx) => removeInputPort(id, idx)}
        />
      ))}

      {/* Add port button */}
      {!locked && (
        <div style={{ padding: '5px 10px' }}>
          <button
            onClick={e => { e.stopPropagation(); addInputPort(id); }}
            title="Add input port"
            style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '2px 8px', borderRadius: 2,
              border: '1px solid var(--border)', background: 'transparent',
              color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
              fontSize: 9, cursor: 'pointer', transition: 'all 0.12s',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--phosphor-dim)'; e.currentTarget.style.color = 'var(--phosphor)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-muted)'; }}
          >
            + add input
          </button>
        </div>
      )}

      {/* Warnings */}
      {data.warnings?.length > 0 && (
        <div style={{ padding: '4px 10px 6px' }}>
          {data.warnings.map((w, i) => (
            <div key={i} style={{
              fontSize: 9, color: 'var(--amber)', fontFamily: 'var(--font-mono)',
              display: 'flex', gap: 4, alignItems: 'flex-start', marginTop: 2,
            }}>
              <span>⚠</span><span>{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default InputLayerNode;