import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';

function MetricChart({ label, data, color = 'var(--phosphor)' }) {
  if (!data || data.length === 0) return null;
  const chartData = data.map((v, i) => ({ i, v: typeof v === 'number' ? v : 0 }));
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: 'var(--text-secondary)',
        marginBottom: 4,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
      }}>
        {label}
      </div>
      <ResponsiveContainer width="100%" height={52}>
        <LineChart data={chartData}>
          <Line
            type="monotone"
            dataKey="v"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <XAxis dataKey="i" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-primary)',
            }}
            formatter={v => [v?.toFixed?.(6) ?? v, '']}
            labelFormatter={() => ''}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ScalarRow({ label, value }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '4px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--phosphor)' }}>
        {typeof value === 'number' ? value.toFixed(6) : JSON.stringify(value)}
      </span>
    </div>
  );
}

export default function MetricsPanel({ node }) {
  if (!node) return (
    <div style={{ padding: 20, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
      Select a layer or node to inspect metrics.
    </div>
  );

  const { data } = node;
  const metrics  = data.metrics;
  const kind     = data.layerType ?? data.nodeType;

  return (
    <div style={{ padding: 16, overflowY: 'auto', height: '100%' }}>
      {/* Node identity */}
      <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: 11,
          color: 'var(--phosphor)',
          letterSpacing: '0.12em',
          marginBottom: 4,
        }}>
          {kind}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
          id: {node.id}
        </div>
        <div style={{
          display: 'inline-block',
          marginTop: 6,
          padding: '2px 8px',
          borderRadius: 2,
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          color: statusColor(data.status),
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          {data.status}
        </div>
      </div>

      {/* Config */}
      {data.config && Object.keys(data.config).length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <SectionLabel>Config</SectionLabel>
          {Object.entries(data.config).map(([k, v]) => (
            <ScalarRow key={k} label={k} value={v} />
          ))}
        </div>
      )}

      {/* Shapes */}
      {metrics?.input_shape && (
        <div style={{ marginBottom: 16 }}>
          <SectionLabel>Shapes</SectionLabel>
          <ScalarRow label="input"  value={metrics.input_shape?.join(' × ')} />
          <ScalarRow label="output" value={metrics.output_shape?.join(' × ')} />
        </div>
      )}

      {/* No metrics yet */}
      {!metrics && (
        <div style={{
          padding: 16,
          background: 'var(--bg-elevated)',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-muted)',
          textAlign: 'center',
        }}>
          No metrics yet — run the network to collect data.
        </div>
      )}

      {/* Scalar metrics */}
      {metrics && (
        <>
          {metrics.activation_mean !== undefined && (
            <div style={{ marginBottom: 16 }}>
              <SectionLabel>Activation</SectionLabel>
              <ScalarRow label="mean" value={metrics.activation_mean} />
              <ScalarRow label="var"  value={metrics.activation_var} />
            </div>
          )}

          {metrics.grad_norm !== undefined && (
            <div style={{ marginBottom: 16 }}>
              <SectionLabel>Gradients</SectionLabel>
              <ScalarRow label="norm" value={metrics.grad_norm} />
              <ScalarRow label="var"  value={metrics.grad_var} />
            </div>
          )}

          {metrics.residual_energy !== undefined && (
            <div style={{ marginBottom: 16 }}>
              <SectionLabel>Path Energy</SectionLabel>
              <ScalarRow label="residual" value={metrics.residual_energy} />
              <ScalarRow label="shortcut" value={metrics.shortcut_energy} />
            </div>
          )}

          {/* Series charts */}
          {metrics.activation_mean_series && (
            <MetricChart
              label="Activation Mean (history)"
              data={metrics.activation_mean_series}
              color="var(--phosphor)"
            />
          )}
          {metrics.grad_norm_series && (
            <MetricChart
              label="Grad Norm (history)"
              data={metrics.grad_norm_series}
              color="var(--amber)"
            />
          )}
        </>
      )}

      {/* Warnings */}
      {data.warnings?.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <SectionLabel color="var(--amber)">Warnings</SectionLabel>
          {data.warnings.map((w, i) => (
            <div key={i} style={{
              padding: '6px 8px',
              marginBottom: 4,
              background: 'rgba(255,170,0,0.05)',
              border: '1px solid rgba(255,170,0,0.2)',
              borderRadius: 3,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--amber)',
            }}>
              ⚠ {w}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SectionLabel({ children, color = 'var(--text-muted)' }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)',
      fontSize: 9,
      color,
      textTransform: 'uppercase',
      letterSpacing: '0.12em',
      marginBottom: 6,
      marginTop: 4,
    }}>
      {children}
    </div>
  );
}

function statusColor(s) {
  return { idle: 'var(--text-muted)', done: 'var(--phosphor)', running: 'var(--phosphor)',
           error: 'var(--red)', pending: 'var(--amber)', locked: 'var(--text-muted)',
           branch: 'var(--blue)' }[s] ?? 'var(--text-muted)';
}