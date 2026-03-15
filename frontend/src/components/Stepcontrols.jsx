import React from 'react';
import { useGraphStore } from '../store/Usegraphstore';
import { useStepSession } from '../hooks/Usewebsocket';

export default function StepControls() {
  const {
    runMode, stepPhase, currentStepId,
    pendingBranches, completedBranches,
    nodes, edges,
    runConfig, inputFile,
  } = useGraphStore();

  const { start, next, prev, followBranch, stop } = useStepSession();

  const isStepping = runMode === 'stepping';
  const isIdle     = runMode === 'idle';
  const isDone     = runMode === 'done';
  const isError    = runMode === 'error';
  const atBranch   = pendingBranches.length > 0;

  const currentNode = nodes.find(n => n.id === currentStepId);
  const graph = { nodes, edges };

  return (
    <div style={{ padding: 16, overflowY: 'auto', height: '100%' }}>

      {/* Status indicator */}
      <div style={{
        padding: '10px 12px',
        background: 'var(--bg-surface)',
        border: `1px solid ${modeColor(runMode)}`,
        borderRadius: 'var(--radius)',
        marginBottom: 16,
        boxShadow: isStepping ? `0 0 16px ${modeColor(runMode)}22` : 'none',
      }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: 9,
          letterSpacing: '0.12em',
          color: modeColor(runMode),
          marginBottom: 4,
        }}>
          {modeLabel(runMode, stepPhase)}
        </div>
        {currentNode && (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
            {currentNode.data.layerType ?? currentNode.data.nodeType} — {currentStepId}
          </div>
        )}
        {!currentNode && isStepping && (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
            awaiting next action
          </div>
        )}
      </div>

      {/* Branch panel */}
      {atBranch && (
        <div style={{ marginBottom: 16 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            color: 'var(--blue)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginBottom: 8,
          }}>
            Branch Point — follow each path
          </div>
          {pendingBranches.map(b => {
            const done = completedBranches.includes(b);
            return (
              <button
                key={b}
                disabled={done}
                onClick={() => followBranch(b)}
                style={{
                  display: 'block',
                  width: '100%',
                  marginBottom: 6,
                  padding: '7px 12px',
                  background: done ? 'var(--bg-surface)' : 'rgba(0,153,255,0.08)',
                  border: `1px solid ${done ? 'var(--border)' : 'var(--blue-dim)'}`,
                  borderRadius: 'var(--radius-sm)',
                  color: done ? 'var(--text-muted)' : 'var(--blue)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  textAlign: 'left',
                  cursor: done ? 'default' : 'pointer',
                  transition: 'all 0.15s',
                }}
              >
                {done ? '✓' : '▶'} {b}
              </button>
            );
          })}
        </div>
      )}

      {/* Main controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

        {/* Start step mode */}
        {(isIdle || isDone || isError) && (
          <CtrlButton
            color="var(--phosphor)"
            onClick={() => start(graph, runConfig, inputFile)}
          >
            ▶ START STEP MODE
          </CtrlButton>
        )}

        {/* Next */}
        {isStepping && stepPhase === 'forward' && !atBranch && (
          <CtrlButton color="var(--phosphor)" onClick={next}>
            NEXT LAYER →
          </CtrlButton>
        )}

        {/* Prev (backward) */}
        {isStepping && stepPhase === 'backward' && (
          <CtrlButton color="var(--amber)" onClick={prev}>
            ← PREV LAYER
          </CtrlButton>
        )}

        {/* Stop */}
        {isStepping && (
          <CtrlButton color="var(--red)" onClick={stop} outline>
            ■ STOP
          </CtrlButton>
        )}

        {/* Done state */}
        {isDone && (
          <div style={{
            padding: '8px 12px',
            background: 'var(--phosphor-faint)',
            border: '1px solid var(--phosphor-dim)',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--phosphor)',
            textAlign: 'center',
          }}>
            ✓ Run complete
          </div>
        )}

        {/* Error state */}
        {isError && (
          <div style={{
            padding: '8px 12px',
            background: 'rgba(255,51,85,0.05)',
            border: '1px solid var(--red-dim)',
            borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--red)',
          }}>
            ✗ Error — check the highlighted layer, correct config, then restart.
          </div>
        )}
      </div>

      {/* Hints */}
      <div style={{ marginTop: 20 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', lineHeight: 1.7 }}>
          {isIdle && '— Select a start node or run from the beginning.\n— All previous layers must have been run at least once to start mid-network.'}
          {isStepping && stepPhase === 'forward' && '— Click NEXT to advance one layer.\n— At branch points, follow all branches before continuing.'}
          {isStepping && stepPhase === 'backward' && '— Click PREV to step backward.\n— Branch paths must all be backpropagated before merging.'}
        </div>
      </div>
    </div>
  );
}

function CtrlButton({ children, color, onClick, outline }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '9px 16px',
        background: outline ? 'transparent' : `${color}14`,
        border: `1.5px solid ${color}`,
        borderRadius: 'var(--radius-sm)',
        color,
        fontFamily: 'var(--font-display)',
        fontSize: 9,
        letterSpacing: '0.12em',
        cursor: 'pointer',
        transition: 'all 0.15s',
        width: '100%',
      }}
      onMouseEnter={e => e.currentTarget.style.background = `${color}26`}
      onMouseLeave={e => e.currentTarget.style.background = outline ? 'transparent' : `${color}14`}
    >
      {children}
    </button>
  );
}

function modeColor(mode) {
  return {
    idle:        'var(--border-bright)',
    connecting:  'var(--amber)',
    stepping:    'var(--phosphor)',
    running:     'var(--phosphor)',
    done:        'var(--phosphor-dim)',
    error:       'var(--red)',
  }[mode] ?? 'var(--border-bright)';
}

function modeLabel(mode, phase) {
  if (mode === 'idle')       return 'IDLE';
  if (mode === 'connecting') return 'CONNECTING...';
  if (mode === 'stepping')   return phase === 'forward' ? 'STEPPING — FORWARD' : 'STEPPING — BACKWARD';
  if (mode === 'running')    return 'RUNNING';
  if (mode === 'done')       return 'COMPLETE';
  if (mode === 'error')      return 'ERROR';
  return mode.toUpperCase();
}