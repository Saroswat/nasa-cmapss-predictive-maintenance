"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Clock3,
  Database,
  Gauge,
  ListFilter,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Wrench,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type HistoryPoint = { cycle: number; predictedRul: number; risk: number };
type Engine = {
  id: number;
  cycle: number;
  actualRul: number;
  predictedRul: number;
  risk: number;
  recommended: boolean;
  actualMaintenance: boolean;
  history: HistoryPoint[];
};

type DashboardData = {
  meta: {
    dataset: string;
    generatedAt: string;
    model: string;
    threshold: number;
    maintenanceHorizon: number;
  };
  metrics: {
    test_regression: { mae: number; rmse: number; r2: number; nasa_score: number };
    test_maintenance_value: {
      expected_value: number;
      false_negatives: number;
      false_positives: number;
      true_negatives: number;
      true_positives: number;
    };
    validation_regression: Record<
      string,
      { mae: number; rmse: number; r2: number; nasa_score: number }
    >;
    feature_count: number;
  };
  featureImportance: { feature: string; importance: number }[];
  engines: Engine[];
};

type View = "overview" | "maintenance" | "model";
type RiskFilter = "all" | "critical" | "watch" | "stable";

const compactCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumFractionDigits: 1,
});

function riskState(risk: number, threshold: number) {
  if (risk >= Math.max(0.65, threshold)) return "critical";
  if (risk >= threshold) return "watch";
  return "stable";
}

function RiskBadge({ risk, threshold }: { risk: number; threshold: number }) {
  const state = riskState(risk, threshold);
  return <span className={`status status-${state}`}>{state}</span>;
}

function Metric({
  label,
  value,
  detail,
  icon: Icon,
  tone = "default",
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Activity;
  tone?: "default" | "warning" | "positive";
}) {
  return (
    <div className={`metric metric-${tone}`}>
      <div className="metric-icon" aria-hidden="true">
        <Icon size={18} />
      </div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <span>{detail}</span>
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="empty-state">
      <ListFilter size={22} />
      <span>{message}</span>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<RiskFilter>("all");
  const [threshold, setThreshold] = useState(0.14);
  const [selected, setSelected] = useState<Engine | null>(null);

  useEffect(() => {
    fetch("/data/dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("Dashboard data is unavailable");
        return response.json();
      })
      .then((payload: DashboardData) => {
        setData(payload);
        setThreshold(payload.meta.threshold);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const fleet = useMemo(() => {
    if (!data) return [];
    return data.engines
      .filter((engine) => {
        const matchesQuery = `ENG-${String(engine.id).padStart(3, "0")}`
          .toLowerCase()
          .includes(query.toLowerCase());
        const state = riskState(engine.risk, threshold);
        return matchesQuery && (filter === "all" || filter === state);
      })
      .sort((a, b) => b.risk - a.risk);
  }, [data, filter, query, threshold]);

  const summary = useMemo(() => {
    if (!data) return { due: 0, critical: 0, meanRul: 0 };
    const due = data.engines.filter((engine) => engine.risk >= threshold).length;
    const critical = data.engines.filter(
      (engine) => riskState(engine.risk, threshold) === "critical",
    ).length;
    const meanRul =
      data.engines.reduce((total, engine) => total + engine.predictedRul, 0) /
      data.engines.length;
    return { due, critical, meanRul };
  }, [data, threshold]);

  const distribution = useMemo(() => {
    if (!data) return [];
    const buckets = [
      { label: "0–20%", min: 0, max: 0.2, count: 0, color: "#2f766d" },
      { label: "20–40%", min: 0.2, max: 0.4, count: 0, color: "#6b8c72" },
      { label: "40–60%", min: 0.4, max: 0.6, count: 0, color: "#c79a38" },
      { label: "60–80%", min: 0.6, max: 0.8, count: 0, color: "#d26945" },
      { label: "80–100%", min: 0.8, max: 1.01, count: 0, color: "#b5473e" },
    ];
    data.engines.forEach((engine) => {
      const bucket = buckets.find((item) => engine.risk >= item.min && engine.risk < item.max);
      if (bucket) bucket.count += 1;
    });
    return buckets;
  }, [data]);

  if (error) {
    return (
      <main className="load-state">
        <AlertTriangle size={28} />
        <h1>Fleet data unavailable</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="load-state" aria-live="polite">
        <Activity className="loading-mark" size={30} />
        <h1>Loading fleet intelligence</h1>
      </main>
    );
  }

  const nav = [
    { id: "overview" as const, label: "Fleet overview", icon: CircleGauge },
    { id: "maintenance" as const, label: "Maintenance", icon: Wrench },
    { id: "model" as const, label: "Model health", icon: BarChart3 },
  ];

  const queue = data.engines
    .filter((engine) => engine.risk >= threshold)
    .sort((a, b) => b.risk - a.risk);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Gauge size={20} /></div>
          <div>
            <strong>AERO<span className="brand-accent">PULSE</span></strong>
            <span className="brand-subtitle">Engine Health Intelligence</span>
          </div>
        </div>
        <nav aria-label="Dashboard views">
          {nav.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={view === id ? "nav-item active" : "nav-item"}
              onClick={() => setView(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-meta">
          <div className="system-state"><span /> Model online</div>
          <p>{data.meta.dataset}</p>
          <span>{data.meta.model}</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Operations / {nav.find((item) => item.id === view)?.label}</p>
            <h1>{view === "overview" ? "Fleet overview" : view === "maintenance" ? "Maintenance queue" : "Model health"}</h1>
          </div>
          <div className="topbar-actions">
            <div className="sync-state"><Database size={15} /><span>100 engines synced</span></div>
          </div>
        </header>

        <section className="threshold-band" aria-label="Decision threshold">
          <div className="threshold-copy">
            <SlidersHorizontal size={18} />
            <div>
              <strong>Maintenance threshold</strong>
              <span>Probability required to add an engine to the queue</span>
            </div>
          </div>
          <input
            aria-label="Maintenance probability threshold"
            type="range"
            min="5"
            max="95"
            step="1"
            value={Math.round(threshold * 100)}
            onChange={(event) => setThreshold(Number(event.target.value) / 100)}
          />
          <output>{Math.round(threshold * 100)}%</output>
        </section>

        {view === "overview" && (
          <>
            <section className="metric-grid" aria-label="Fleet metrics">
              <Metric icon={Activity} label="Fleet assessed" value="100" detail="FD001 test engines" />
              <Metric icon={Wrench} label="Maintenance queue" value={String(summary.due)} detail={`${summary.critical} critical priority`} tone="warning" />
              <Metric icon={Clock3} label="Mean predicted RUL" value={`${summary.meanRul.toFixed(1)}`} detail="remaining cycles" />
              <Metric icon={ShieldCheck} label="Expected value" value={compactCurrency.format(data.metrics.test_maintenance_value.expected_value)} detail="verified test policy" tone="positive" />
            </section>

            <section className="overview-grid">
              <div className="panel chart-panel">
                <div className="panel-heading">
                  <div><h2>Fleet risk distribution</h2><p>Maintenance probability across the test fleet</p></div>
                  <span className="data-chip">100 engines</span>
                </div>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={distribution} margin={{ top: 8, right: 4, left: -24, bottom: 0 }}>
                      <CartesianGrid stroke="#e7e9e8" vertical={false} />
                      <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={12} />
                      <YAxis tickLine={false} axisLine={false} fontSize={12} allowDecimals={false} />
                      <Tooltip cursor={{ fill: "#f3f5f4" }} contentStyle={{ borderRadius: 6, borderColor: "#d9dddb" }} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {distribution.map((entry) => <Cell key={entry.label} fill={entry.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel policy-panel">
                <div className="panel-heading">
                  <div><h2>Policy outcome</h2><p>30-cycle maintenance horizon</p></div>
                  <CheckCircle2 size={20} className="positive-icon" />
                </div>
                <div className="outcome-value">
                  <span>Expected fleet value</span>
                  <strong>{compactCurrency.format(data.metrics.test_maintenance_value.expected_value)}</strong>
                </div>
                <div className="outcome-grid">
                  <div><span>True positive</span><strong>{data.metrics.test_maintenance_value.true_positives}</strong></div>
                  <div><span>True negative</span><strong>{data.metrics.test_maintenance_value.true_negatives}</strong></div>
                  <div><span>False positive</span><strong>{data.metrics.test_maintenance_value.false_positives}</strong></div>
                  <div><span>False negative</span><strong>{data.metrics.test_maintenance_value.false_negatives}</strong></div>
                </div>
              </div>
            </section>

            <section className="panel fleet-panel">
              <div className="fleet-toolbar">
                <div><h2>Engine fleet</h2><p>{fleet.length} engines in current view</p></div>
                <div className="toolbar-controls">
                  <label className="search-box">
                    <Search size={16} />
                    <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search engine" aria-label="Search engines" />
                  </label>
                  <div className="segments" aria-label="Risk filter">
                    {(["all", "critical", "watch", "stable"] as RiskFilter[]).map((item) => (
                      <button key={item} onClick={() => setFilter(item)} className={filter === item ? "active" : ""}>{item}</button>
                    ))}
                  </div>
                </div>
              </div>
              <FleetTable engines={fleet} threshold={threshold} onSelect={setSelected} />
            </section>
          </>
        )}

        {view === "maintenance" && (
          <section className="maintenance-layout">
            <div className="panel queue-panel">
              <div className="panel-heading">
                <div><h2>Prioritized work queue</h2><p>{queue.length} engines at or above {Math.round(threshold * 100)}% risk</p></div>
                <span className="priority-count">{summary.critical} critical</span>
              </div>
              {queue.length ? <FleetTable engines={queue} threshold={threshold} onSelect={setSelected} compact /> : <EmptyState message="No engines meet the current threshold" />}
            </div>
            <aside className="maintenance-summary">
              <div className="panel">
                <div className="panel-heading"><div><h2>Decision matrix</h2><p>Verified FD001 outcomes</p></div></div>
                <div className="matrix" aria-label="Confusion matrix">
                  <div className="matrix-cell tn"><strong>{data.metrics.test_maintenance_value.true_negatives}</strong><span>True negative</span></div>
                  <div className="matrix-cell fp"><strong>{data.metrics.test_maintenance_value.false_positives}</strong><span>False positive</span></div>
                  <div className="matrix-cell fn"><strong>{data.metrics.test_maintenance_value.false_negatives}</strong><span>False negative</span></div>
                  <div className="matrix-cell tp"><strong>{data.metrics.test_maintenance_value.true_positives}</strong><span>True positive</span></div>
                </div>
              </div>
            </aside>
          </section>
        )}

        {view === "model" && (
          <>
            <section className="metric-grid model-metrics">
              <Metric icon={Gauge} label="MAE" value={data.metrics.test_regression.mae.toFixed(2)} detail="cycles" />
              <Metric icon={Activity} label="RMSE" value={data.metrics.test_regression.rmse.toFixed(2)} detail="cycles" />
              <Metric icon={BarChart3} label="R²" value={data.metrics.test_regression.r2.toFixed(3)} detail="official test set" tone="positive" />
              <Metric icon={ShieldCheck} label="NASA score" value={data.metrics.test_regression.nasa_score.toFixed(2)} detail="lower is better" />
            </section>
            <section className="model-grid">
              <div className="panel chart-panel">
                <div className="panel-heading"><div><h2>Validation comparison</h2><p>Held-out engine checkpoints</p></div><span className="data-chip">Selected: RF</span></div>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(data.metrics.validation_regression).map(([name, values]) => ({ name: name === "random_forest" ? "Random forest" : "Hist. boosting", MAE: values.mae, RMSE: values.rmse }))} margin={{ top: 14, right: 8, left: -16, bottom: 0 }}>
                      <CartesianGrid stroke="#e7e9e8" vertical={false} />
                      <XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={12} />
                      <YAxis tickLine={false} axisLine={false} fontSize={12} />
                      <Tooltip contentStyle={{ borderRadius: 6, borderColor: "#d9dddb" }} />
                      <Bar dataKey="MAE" fill="#287271" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="RMSE" fill="#d0a03c" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="panel feature-panel">
                <div className="panel-heading"><div><h2>Feature influence</h2><p>Top random forest signals</p></div><span className="data-chip">{data.metrics.feature_count} total</span></div>
                <div className="feature-list">
                  {data.featureImportance.map((item, index) => (
                    <div className="feature-row" key={item.feature}>
                      <span>{index + 1}</span>
                      <div><strong>{item.feature.replaceAll("_", " ")}</strong><div className="feature-track"><i style={{ width: `${(item.importance / data.featureImportance[0].importance) * 100}%` }} /></div></div>
                      <em>{(item.importance * 100).toFixed(1)}%</em>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </>
        )}
      </main>

      {selected && <EngineDrawer engine={selected} threshold={threshold} onClose={() => setSelected(null)} />}
    </div>
  );
}

function FleetTable({ engines, threshold, onSelect, compact = false }: { engines: Engine[]; threshold: number; onSelect: (engine: Engine) => void; compact?: boolean }) {
  if (!engines.length) return <EmptyState message="No engines match the current filters" />;
  return (
    <div className={compact ? "table-wrap compact" : "table-wrap"}>
      <table>
        <thead><tr><th>Engine</th><th>Current cycle</th><th>Predicted RUL</th><th>Maintenance risk</th><th>Decision</th><th><span className="sr-only">Open</span></th></tr></thead>
        <tbody>
          {engines.map((engine) => (
            <tr key={engine.id} onClick={() => onSelect(engine)} tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter") onSelect(engine); }}>
              <td><strong>ENG-{String(engine.id).padStart(3, "0")}</strong><span className="row-sub">FD001 fleet</span></td>
              <td>{engine.cycle}</td>
              <td><strong>{engine.predictedRul.toFixed(1)}</strong><span className="row-sub">cycles</span></td>
              <td><div className="risk-cell"><div className="risk-track"><i style={{ width: `${Math.max(engine.risk * 100, 1)}%` }} /></div><span>{Math.round(engine.risk * 100)}%</span></div></td>
              <td><RiskBadge risk={engine.risk} threshold={threshold} /></td>
              <td><button className="row-action" aria-label={`Open engine ${engine.id}`}><ChevronRight size={17} /></button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EngineDrawer({ engine, threshold, onClose }: { engine: Engine; threshold: number; onClose: () => void }) {
  const state = riskState(engine.risk, threshold);
  const [scheduled, setScheduled] = useState(false);
  return (
    <div className="drawer-layer" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) onClose(); }}>
      <aside className="engine-drawer" role="dialog" aria-modal="true" aria-label={`Engine ${engine.id} details`}>
        <header><div><p>FD001 / ENGINE</p><h2>ENG-{String(engine.id).padStart(3, "0")}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close engine details"><X size={19} /></button></header>
        <div className={`drawer-alert drawer-alert-${state}`}><AlertTriangle size={18} /><div><strong>{state === "stable" ? "No action required" : state === "critical" ? "Immediate review" : "Maintenance review"}</strong><span>{Math.round(engine.risk * 100)}% predicted maintenance risk</span></div></div>
        <div className="drawer-stats">
          <div><span>Predicted RUL</span><strong>{engine.predictedRul.toFixed(1)}</strong><em>cycles</em></div>
          <div><span>Current cycle</span><strong>{engine.cycle}</strong><em>observed</em></div>
          <div><span>Policy threshold</span><strong>{Math.round(threshold * 100)}%</strong><em>probability</em></div>
        </div>
        <section className="drawer-chart"><div><h3>Model trajectory</h3><p>Last {engine.history.length} observed cycles</p></div><div className="drawer-chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={engine.history} margin={{ top: 10, right: 12, left: -22, bottom: 0 }}><CartesianGrid stroke="#e7e9e8" vertical={false} /><XAxis dataKey="cycle" tickLine={false} axisLine={false} fontSize={11} /><YAxis tickLine={false} axisLine={false} fontSize={11} /><Tooltip contentStyle={{ borderRadius: 6, borderColor: "#d9dddb" }} /><Line type="monotone" dataKey="predictedRul" name="Predicted RUL" stroke="#287271" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></div></section>
        <section className="risk-history"><div><h3>Risk trend</h3><p>Model probability</p></div><div className="risk-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={engine.history} margin={{ top: 8, right: 8, left: -26, bottom: 0 }}><CartesianGrid stroke="#e7e9e8" vertical={false} /><XAxis dataKey="cycle" hide /><YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} fontSize={11} /><Tooltip formatter={(value) => `${Math.round(Number(value) * 100)}%`} contentStyle={{ borderRadius: 6, borderColor: "#d9dddb" }} /><Area type="monotone" dataKey="risk" stroke="#b5473e" fill="#f2d9d6" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></section>
        <footer><button className="secondary-button" onClick={onClose}>Close</button><button className="primary-button" onClick={() => setScheduled(true)} disabled={scheduled}>{scheduled ? <CheckCircle2 size={16} /> : <Wrench size={16} />}{scheduled ? "Scheduled" : "Add to maintenance"}</button></footer>
      </aside>
    </div>
  );
}
