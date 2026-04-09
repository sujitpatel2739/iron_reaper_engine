import React, { useCallback, useRef } from 'react';
import ReactFlow, {
  Background, Controls, MiniMap,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';

import LayerNode      from './Layernode';
import OperationNode  from './Operationnode';
import InputLayerNode from './Inputlayernode';
import { useGraphStore } from '../store/Usegraphstore';

const nodeTypes = {
  layerNode:      LayerNode,
  operationNode:  OperationNode,
  inputLayerNode: InputLayerNode,
};

export default function GraphCanvas() {
  const {
    nodes, edges,
    onNodesChange, onEdgesChange, onConnect,
    setSelectedNode, runMode,
    layoutDirection,
  } = useGraphStore();

  const locked = runMode !== 'idle';

  // -- Drag offset tracking ------------------------------------------------
  // Store where inside the pill the user started dragging so the dropped node
  // lands exactly where the cursor released, not offset by a fixed amount.
  const dragOffset = useRef({ x: 0, y: 0 });

  // Called on the draggable pill in Toolbar — captures cursor offset inside element
  const onDragStartCapture = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    dragOffset.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }, []);

  // Expose setter so Toolbar pills can attach the handler
  // (passed via dataTransfer as a side-channel isn't possible, so we store globally)
  React.useEffect(() => {
    window.__setDragOffset = (x, y) => { dragOffset.current = { x, y }; };
    return () => { delete window.__setDragOffset; };
  }, []);

  // -- Drop ----------------------------------------------------------------
  const onDrop = useCallback((e) => {
    e.preventDefault();
    if (locked) return;

    const type   = e.dataTransfer.getData('nodeType');
    const kind   = e.dataTransfer.getData('nodeKind');
    const bounds = e.currentTarget.getBoundingClientRect();

    // Subtract the offset where the user grabbed the element so the node's
    // top-left corner lands exactly at the release point
    const position = {
      x: e.clientX - bounds.left - dragOffset.current.x,
      y: e.clientY - bounds.top  - dragOffset.current.y,
    };

    const store = useGraphStore.getState();
    if (kind === 'layer')      store.addLayer(type, position);
    if (kind === 'operation')  store.addOperationNode(type, position);
    if (kind === 'inputLayer') store.addInputLayer(position);
  }, [locked]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onPaneClick = useCallback(() => setSelectedNode(null), [setSelectedNode]);

  // -- One edge per target handle ------------------------------------------
  // Reject a new connection if the target handle already has an incoming edge.
  // Source handles can fan out freely — only target is restricted.
  const isValidConnection = useCallback((connection) => {
    const { target, targetHandle } = connection;
    const existingEdges = useGraphStore.getState().edges;
    const alreadyConnected = existingEdges.some(
      e => e.target === target && e.targetHandle === targetHandle
    );
    return !alreadyConnected;
  }, []);

  return (
    <div style={{ flex: 1, position: 'relative', background: 'var(--bg-void)' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={locked ? undefined : onNodesChange}
        onEdgesChange={locked ? undefined : onEdgesChange}
        onConnect={locked ? undefined : (conn) => {
          if (isValidConnection(conn)) onConnect(conn);
        }}
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
            if (n.data?.kind === 'inputLayer') return 'var(--phosphor)';
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
            fontFamily: 'var(--font-mono)', fontSize: 10,
            color: 'var(--text-muted)', background: 'var(--bg-panel)',
            padding: '3px 8px', borderRadius: 3, border: '1px solid var(--border)',
          }}>
            GRAPH LOCKED — stop run to edit
          </span>
        </div>
      )}
    </div>
  );
}