import React, { useCallback, useRef } from 'react';
import ReactFlow, {
  Background, Controls, MiniMap,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';

import LayerNode from './Layernode';
import OperationNode from './Operationnode';
import { useGraphStore } from '../store/Usegraphstore';

const nodeTypes = {
  layerNode:     LayerNode,
  operationNode: OperationNode,
};

export default function GraphCanvas() {
  const {
    nodes, edges,
    onNodesChange, onEdgesChange, onConnect,
    setSelectedNode, runMode, layoutDirection,
    addLayer,
  } = useGraphStore();

  const locked = runMode !== 'idle';

  // Drop from toolbar palette
  const onDrop = useCallback((e) => {
    e.preventDefault();
    if (locked) return;
    const type     = e.dataTransfer.getData('nodeType');
    const kind     = e.dataTransfer.getData('nodeKind');   // 'layer' | 'operation'
    const bounds   = e.currentTarget.getBoundingClientRect();
    const position = { x: e.clientX - bounds.left - 80, y: e.clientY - bounds.top - 30 };
    if (kind === 'layer')     useGraphStore.getState().addLayer(type, position);
    if (kind === 'operation') useGraphStore.getState().addOperationNode(type, position);
  }, [locked]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onPaneClick = useCallback(() => setSelectedNode(null), []);

  return (
    <div style={{ flex: 1, position: 'relative', background: 'var(--bg-void)' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={locked ? undefined : onNodesChange}
        onEdgesChange={locked ? undefined : onEdgesChange}
        onConnect={locked ? undefined : onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onPaneClick={onPaneClick}
        fitView
        deleteKeyCode={locked ? null : 'Backspace'}
        nodesDraggable={!locked}
        nodesConnectable={!locked}
        elementsSelectable={!locked}
        style={{ background: 'var(--bg-void)' }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="var(--border)"
          className="rf-background-dots"
        />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={n => {
            const s = n.data?.status;
            if (s === 'done')    return 'var(--phosphor-dim)';
            if (s === 'error')   return 'var(--red)';
            if (s === 'running') return 'var(--phosphor)';
            if (s === 'pending') return 'var(--amber)';
            return 'var(--border-bright)';
          }}
          maskColor="rgba(0,0,0,0.7)"
        />
      </ReactFlow>

      {/* Locked overlay */}
      {locked && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(0,0,0,0.15)',
          pointerEvents: 'none',
          zIndex: 5,
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'flex-start',
          padding: 12,
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-muted)',
            background: 'var(--bg-panel)',
            padding: '3px 8px',
            borderRadius: 3,
            border: '1px solid var(--border)',
          }}>
            GRAPH LOCKED — stop run to edit
          </span>
        </div>
      )}
    </div>
  );
}
