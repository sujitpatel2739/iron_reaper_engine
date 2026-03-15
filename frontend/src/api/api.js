const BASE = 'http://localhost:8000';

// ── REST ────────────────────────────────────────────────────────────────────

export async function fetchLayerTypes() {
  const r = await fetch(`${BASE}/layers`);
  return r.json();
}

export async function fetchNodeTypes() {
  const r = await fetch(`${BASE}/nodes`);
  return r.json();
}

export async function validateNetwork(graph) {
  const r = await fetch(`${BASE}/network/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(graph),
  });
  return r.json();  // { valid, warnings: [{node_id, message}] }
}

export async function importModel(file) {
  const fd = new FormData();
  fd.append('model_file', file);
  const r = await fetch(`${BASE}/network/import`, { method: 'POST', body: fd });
  return r.json();  // full graph JSON
}

export async function saveNetwork(graph) {
  const r = await fetch(`${BASE}/network/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(graph),
  });
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'network.json';
  a.click();
  URL.revokeObjectURL(url);
}

export async function loadNetwork(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => {
      try { resolve(JSON.parse(e.target.result)); }
      catch { reject(new Error('Invalid network file')); }
    };
    reader.readAsText(file);
  });
}

export async function runFull(graph, runConfig, inputFile) {
  const fd = new FormData();
  fd.append('graph', JSON.stringify(graph));
  fd.append('run_config', JSON.stringify(runConfig));
  if (inputFile) fd.append('input_file', inputFile);
  const r = await fetch(`${BASE}/run/start`, { method: 'POST', body: fd });
  return r.json();  // DiagnosticReport
}

// ── WebSocket (step mode) ───────────────────────────────────────────────────

export function createStepSession(handlers) {
  const ws = new WebSocket(`ws://localhost:8000/run/step`);

  ws.onopen    = ()      => handlers.onOpen?.();
  ws.onclose   = ()      => handlers.onClose?.();
  ws.onerror   = (e)     => handlers.onError?.(e);
  ws.onmessage = (e)     => {
    try {
      const msg = JSON.parse(e.data);
      handlers.onMessage?.(msg);
    } catch {}
  };

  return {
    send: (action, payload = {}) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action, ...payload }));
      }
    },
    close: () => ws.close(),
  };
}
