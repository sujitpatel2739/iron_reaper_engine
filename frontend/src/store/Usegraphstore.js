import { create } from 'zustand';
import { addEdge, applyNodeChanges, applyEdgeChanges } from 'reactflow';

const DEFAULT_LAYER_TYPES = {
  Linear:    { fields: [{ name: 'in_features', type: 'int', min: 1, default: 128 }, { name: 'out_features', type: 'int', min: 1, default: 64 }] },
  Relu:      { fields: [] },
  LayerNorm: { fields: [{ name: 'in_features', type: 'int', min: 1, default: 64 }, { name: 'eps', type: 'float', default: 1e-5 }] },
};

const DEFAULT_NODE_TYPES = {
  AddNode:    { fields: [{ name: 'axis', type: 'int_or_null', default: null }], inputs: 'N' },
  SubNode:    { fields: [], inputs: 2 },
  MulNode:    { fields: [], inputs: 'N' },
  DivNode:    { fields: [], inputs: 2 },
  SqNode:     { fields: [], inputs: 1 },
  NegNode:    { fields: [], inputs: 1 },
  SqrtNode:   { fields: [], inputs: 1 },
  ScaleNode:  { fields: [{ name: 'scalar', type: 'float', default: 1.0 }], inputs: 1 },
  ClipNode:   { fields: [{ name: 'min_val', type: 'float', default: -1.0 }, { name: 'max_val', type: 'float', default: 1.0 }], inputs: 1 },
  ConcatNode: { fields: [{ name: 'axis', type: 'int', default: 1 }], inputs: 'N' },
  SplitNode:  { fields: [{ name: 'n_splits', type: 'int', min: 2, default: 2 }, { name: 'axis', type: 'int', default: 1 }], inputs: 1 },
};

let _nodeCounter = 0;
function newId() { return `node_${++_nodeCounter}`; }

function _freshLabel(existingInputs) {
  let i = existingInputs.length;
  let candidate = `input_${i}`;
  const existing = new Set(existingInputs);
  while (existing.has(candidate)) { i++; candidate = `input_${i}`; }
  return candidate;
}

export const useGraphStore = create((set, get) => ({
  // ── Graph ──────────────────────────────────────────────────────────────
  nodes: [],
  edges: [],

  onNodesChange: (changes) =>
    set(s => ({ nodes: applyNodeChanges(changes, s.nodes) })),

  onEdgesChange: (changes) =>
    set(s => ({ edges: applyEdgeChanges(changes, s.edges) })),

  onConnect: (conn) =>
    set(s => ({ edges: addEdge({ ...conn, animated: false }, s.edges) })),

  // -- Layer node ----------------------------------------------------------
  addLayer: (type, position = { x: 200, y: 200 }) => {
    const id = newId();
    const typeDef = get().layerTypes[type] ?? DEFAULT_LAYER_TYPES[type] ?? { fields: [] };
    const config  = Object.fromEntries(typeDef.fields.map(f => [f.name, f.default ?? '']));
    set(s => ({
      nodes: [...s.nodes, {
        id,
        type: 'layerNode',
        position,
        data: { kind: 'layer', layerType: type, config, status: 'idle', metrics: null, warnings: [] },
      }],
    }));
    return id;
  },

  // -- Operation node ------------------------------------------------------
  addOperationNode: (type, position = { x: 300, y: 300 }) => {
    const id = newId();
    const typeDef = get().nodeTypes[type] ?? DEFAULT_NODE_TYPES[type] ?? { fields: [] };
    const config  = Object.fromEntries(typeDef.fields.map(f => [f.name, f.default ?? '']));
    set(s => ({
      nodes: [...s.nodes, {
        id,
        type: 'operationNode',
        position,
        data: { kind: 'node', nodeType: type, config, status: 'idle', metrics: null, warnings: [] },
      }],
    }));
    return id;
  },

  // -- InputLayer node -----------------------------------------------------
  addInputLayer: (position = { x: 100, y: 100 }) => {
    const id = newId();
    set(s => ({
      nodes: [...s.nodes, {
        id,
        type: 'inputLayerNode',
        position,
        data: {
          kind:     'inputLayer',
          name:     'inputlayer',
          inputs:   ['input_0'],
          status:   'idle',
          metrics:  null,
          warnings: [],
        },
      }],
    }));
    return id;
  },

  // -- InputLayer port management ------------------------------------------

  addInputPort: (nodeId) =>
    set(s => ({
      nodes: s.nodes.map(n => {
        if (n.id !== nodeId) return n;
        const cur      = n.data.inputs ?? [];
        const newLabel = _freshLabel(cur);
        return { ...n, data: { ...n.data, inputs: [...cur, newLabel] } };
      }),
    })),

  removeInputPort: (nodeId, portIndex) =>
    set(s => ({
      nodes: s.nodes.map(n => {
        if (n.id !== nodeId) return n;
        const cur = n.data.inputs ?? [];
        if (cur.length <= 1) return n;
        return { ...n, data: { ...n.data, inputs: cur.filter((_, i) => i !== portIndex) } };
      }),
    })),

  updateInputLabel: (nodeId, portIndex, newLabel) =>
    set(s => ({
      nodes: s.nodes.map(n => {
        if (n.id !== nodeId) return n;
        const cur   = [...(n.data.inputs ?? [])];
        const clash = cur.some((l, i) => i !== portIndex && l === newLabel);
        if (clash || !newLabel.trim()) return n;
        cur[portIndex] = newLabel.trim();
        return { ...n, data: { ...n.data, inputs: cur } };
      }),
    })),

  updateNodeName: (nodeId, newName) =>
    set(s => ({
      nodes: s.nodes.map(n =>
        n.id !== nodeId ? n : { ...n, data: { ...n.data, name: newName } }
      ),
    })),

  // -- Generic node ops ----------------------------------------------------

  updateNodeConfig: (id, field, value) =>
    set(s => ({
      nodes: s.nodes.map(n => n.id !== id ? n : {
        ...n,
        data: { ...n.data, config: { ...n.data.config, [field]: value } },
      }),
    })),

  setNodeStatus: (id, status) =>
    set(s => ({
      nodes: s.nodes.map(n => n.id !== id ? n : { ...n, data: { ...n.data, status } }),
    })),

  setNodeMetrics: (id, metrics) =>
    set(s => ({
      nodes: s.nodes.map(n => n.id !== id ? n : { ...n, data: { ...n.data, metrics } }),
    })),

  setNodeWarnings: (id, warnings) =>
    set(s => ({
      nodes: s.nodes.map(n => n.id !== id ? n : { ...n, data: { ...n.data, warnings } }),
    })),

  setAllNodeStatuses: (status) =>
    set(s => ({ nodes: s.nodes.map(n => ({ ...n, data: { ...n.data, status } })) })),

  resetMetrics: () =>
    set(s => ({
      nodes: s.nodes.map(n => ({ ...n, data: { ...n.data, metrics: null, status: 'idle' } })),
    })),

  loadGraph: (graph) => {
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
      console.error('loadGraph: invalid graph payload', graph);
      return;
    }
    _nodeCounter = 0;
    set({ nodes: graph.nodes, edges: graph.edges });
  },

  clearGraph: () => set({ nodes: [], edges: [] }),

  // ── Selection ──────────────────────────────────────────────────────────
  selectedNodeId: null,
  setSelectedNode: (id) => set({ selectedNodeId: id }),

  // ── Layout ─────────────────────────────────────────────────────────────
  // Default LR — horizontal flow, nodes spread left to right
  // Toggle swaps x/y of all nodes (handled in Toolbar.handleToggleLayout)
  layoutDirection: 'LR',
  toggleLayout: () =>
    set(s => ({ layoutDirection: s.layoutDirection === 'LR' ? 'TB' : 'LR' })),

  // ── Run state ──────────────────────────────────────────────────────────
  runMode: 'idle',
  stepPhase: 'forward',
  currentStepId: null,
  pendingBranches: [],
  completedBranches: [],
  stepSession: null,

  setRunMode:           (mode)    => set({ runMode: mode }),
  setStepPhase:         (phase)   => set({ stepPhase: phase }),
  setCurrentStep:       (id)      => set({ currentStepId: id }),
  setPendingBranches:   (bs)      => set({ pendingBranches: bs }),
  setCompletedBranches: (bs)      => set({ completedBranches: bs }),
  setStepSession:       (session) => set({ stepSession: session }),

  // ── Run config ─────────────────────────────────────────────────────────
  runConfig: {
    run_id: 0,
    input_shape: [32, 128],
    observers: ['SignalStatsObserver'],
    seed: 42,
  },
  inputFile: null,
  setRunConfig: (cfg) => set(s => ({ runConfig: { ...s.runConfig, ...cfg } })),
  setInputFile: (f)   => set({ inputFile: f }),

  // ── Type registries ────────────────────────────────────────────────────
  layerTypes: DEFAULT_LAYER_TYPES,
  nodeTypes:  DEFAULT_NODE_TYPES,
  setLayerTypes: (t) => set({ layerTypes: t }),
  setNodeTypes:  (t) => set({ nodeTypes: t }),
}));