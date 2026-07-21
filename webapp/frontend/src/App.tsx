import { useEffect, useMemo, useState } from "react";
import type { EChartsCoreOption } from "echarts";
import { apiGet } from "./api";
import EChart from "./components/EChart";
import type { ComparisonModel, ForecastPayload, Meta } from "./types";

type Tab = "forecast" | "signal" | "comparison" | "basis" | "drivers" | "reliability";

const TABS: { id: Tab; label: string }[] = [
  { id: "forecast", label: "Giá & fan chart" },
  { id: "signal", label: "Tín hiệu LONG/FLAT" },
  { id: "comparison", label: "So sánh 22 model" },
  { id: "basis", label: "Nội địa vs London" },
  { id: "drivers", label: "Biến vĩ mô" },
  { id: "reliability", label: "Độ tin cậy" },
];

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"];

function formatVnd(value: number) {
  return `${new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 }).format(value)} VND/kg`;
}

function Loading() {
  return <div className="state">Đang tải dữ liệu…</div>;
}

function ErrorBox({ message }: { message: string }) {
  return <div className="error">Không tải được dữ liệu: {message}</div>;
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <article className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </article>
  );
}

function ForecastPanel({ horizon }: { horizon: number }) {
  const [data, setData] = useState<ForecastPayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setData(null);
    setError("");
    apiGet<ForecastPayload>(`/forecast?horizon=${horizon}&model=best&history_days=180`)
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, [horizon]);

  const option = useMemo<EChartsCoreOption>(() => {
    if (!data) return {};
    const history = data.history.map((item) => [item.date, item.value]);
    const future = data.forecast;
    const band = (level: 50 | 80 | 95, color: string, opacity: number) => [
      {
        name: `Lower ${level}%`,
        type: "line",
        stack: `band-${level}`,
        data: future.map((item) => [
          item.date,
          Number((item as unknown as Record<string, number>)[`lower_${level}`]),
        ]),
        symbol: "none",
        lineStyle: { opacity: 0 },
        tooltip: { show: false },
      },
      {
        name: `${level}% interval`,
        type: "line",
        stack: `band-${level}`,
        data: future.map((item) => {
          const values = item as unknown as Record<string, number>;
          return [item.date, Number(values[`upper_${level}`]) - Number(values[`lower_${level}`])];
        }),
        symbol: "none",
        lineStyle: { opacity: 0 },
        areaStyle: { color, opacity },
        tooltip: { show: false },
      },
    ];
    return {
      animation: false,
      tooltip: { trigger: "axis" },
      legend: { data: ["Lịch sử", "Dự báo", "50% interval", "80% interval", "95% interval"] },
      grid: { left: 70, right: 28, top: 55, bottom: 55 },
      xAxis: { type: "time" },
      yAxis: { type: "value", name: "VND/kg", scale: true },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 18 }],
      series: [
        ...band(95, "#56B4E9", 0.14),
        ...band(80, "#0072B2", 0.18),
        ...band(50, "#009E73", 0.24),
        { name: "Lịch sử", type: "line", data: history, symbol: "none", lineStyle: { color: "#5a3b2a", width: 2 } },
        {
          name: "Dự báo",
          type: "line",
          data: future.map((item) => [item.date, item.forecast]),
          symbol: "circle",
          symbolSize: 5,
          lineStyle: { color: "#D55E00", width: 3, type: "dashed" },
        },
      ],
    };
  }, [data]);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading />;
  const change = (data.point_forecast / data.anchor_price - 1) * 100;
  return (
    <>
      <div className="metric-grid four">
        <Metric label="Giá tham chiếu" value={formatVnd(data.anchor_price)} note={`Dữ liệu đến ${data.data_as_of}`} />
        <Metric label="Giá dự báo" value={formatVnd(data.point_forecast)} note={`${change >= 0 ? "+" : ""}${change.toFixed(2)}%`} />
        <Metric label="Mô hình" value={data.model} note={`h=${data.horizon} phiên`} />
        <Metric label="RMSE kiểm thử" value={formatVnd(data.rmse)} note={data.source} />
      </div>
      <section className="panel chart-panel">
        <h2>Giá Robusta nội địa và dải dự báo 50/80/95%</h2>
        <EChart option={option} height={480} />
      </section>
      <p className="disclaimer">{data.disclaimer} Dải bất định rộng dần theo chân trời.</p>
    </>
  );
}

function SignalPanel() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    apiGet("/signal?threshold_pct=1").then(setData).catch((reason: Error) => setError(reason.message));
  }, []);
  if (error) return <ErrorBox message={error} />;
  if (!data) return <Loading />;
  return (
    <section className="signal-layout">
      <div className={`signal-card ${data.signal.toLowerCase()}`}>
        <span>Tín hiệu h=5</span>
        <strong>{data.signal}</strong>
        <small>{data.change_pct >= 0 ? "+" : ""}{data.change_pct.toFixed(2)}%</small>
      </div>
      <article className="panel explanation">
        <h2>Quy tắc quyết định</h2>
        <p><b>LONG</b> khi mức tăng dự báo 5 phiên ≥ {data.threshold_pct}%.</p>
        <p><b>FLAT</b> khi tín hiệu yếu hơn ngưỡng hoặc chỉ có RandomWalk fallback.</p>
        <p className="reason">{data.reason}</p>
        <p className="disclaimer">{data.disclaimer}</p>
      </article>
    </section>
  );
}

function ComparisonPanel({ horizon }: { horizon: number }) {
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setPayload(null);
    apiGet(`/model-comparison?horizon=${horizon}`).then(setPayload).catch((reason: Error) => setError(reason.message));
  }, [horizon]);
  if (error) return <ErrorBox message={error} />;
  if (!payload) return <Loading />;
  const rows = (payload.models as ComparisonModel[]).filter((row) => row.mse !== null);
  const sorted = [...rows].sort((a, b) => (a.delta_vs_rw_pct ?? 0) - (b.delta_vs_rw_pct ?? 0));
  const option: EChartsCoreOption = {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 120, right: 35, top: 20, bottom: 40 },
    xAxis: { type: "value", name: "Δ% so với RandomWalk" },
    yAxis: { type: "category", data: sorted.map((row) => row.model), inverse: true },
    series: [{
      type: "bar",
      data: sorted.map((row) => ({
        value: row.delta_vs_rw_pct,
        itemStyle: { color: (row.delta_vs_rw_pct ?? 0) <= 0 ? "#009E73" : "#D55E00" },
      })),
    }],
  };
  return (
    <div className="two-column">
      <section className="panel">
        <h2>22-model comparison — h={horizon}</h2>
        <EChart option={option} height={620} />
      </section>
      <section className="panel table-panel">
        <h2>MSE ({payload.unit})</h2>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Family</th><th>Model</th><th>MSE</th><th>Δ vs RW</th></tr></thead>
            <tbody>{sorted.map((row) => (
              <tr key={row.model}>
                <td>{row.family}</td><td>{row.model}</td>
                <td>{row.mse?.toFixed(2)}</td>
                <td className={(row.delta_vs_rw_pct ?? 0) <= 0 ? "good" : "bad"}>{row.delta_vs_rw_pct?.toFixed(2)}%</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="disclaimer">{payload.interpretation}</p>
      </section>
    </div>
  );
}

function BasisPanel() {
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => { apiGet("/basis?days=365").then(setPayload).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <ErrorBox message={error} />;
  if (!payload) return <Loading />;
  const fields = ["domestic", "london", "basis"];
  const labels = ["Nội địa", "London quy đổi", "Basis"];
  const option: EChartsCoreOption = {
    tooltip: { trigger: "axis" }, legend: { data: labels },
    grid: { left: 75, right: 25, top: 50, bottom: 55 },
    xAxis: { type: "time" }, yAxis: { type: "value", name: payload.unit, scale: true },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18 }],
    series: fields.map((field, index) => ({
      name: labels[index], type: "line", symbol: "none",
      data: payload.data.map((row: any) => [row.date, row[field]]),
      lineStyle: { color: COLORS[index], width: index === 2 ? 2 : 2.5 },
    })),
  };
  return <section className="panel"><h2>Giá nội địa, London quy đổi và basis</h2><EChart option={option} height={540} /></section>;
}

function DriversPanel() {
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => { apiGet("/drivers?days=730").then(setPayload).catch((e: Error) => setError(e.message)); }, []);
  if (error) return <ErrorBox message={error} />;
  if (!payload) return <Loading />;
  const option: EChartsCoreOption = {
    tooltip: { trigger: "axis" }, legend: { data: payload.drivers },
    grid: { left: 65, right: 25, top: 80, bottom: 55 },
    xAxis: { type: "time" }, yAxis: { type: "value", name: "z-score" },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18 }],
    series: payload.drivers.map((driver: string, index: number) => ({
      name: driver, type: "line", symbol: "none",
      data: payload.data.map((row: any) => [row.date, row[driver]]),
      lineStyle: { color: COLORS[index % COLORS.length], width: 2 },
    })),
  };
  return (
    <section className="panel">
      <h2>Macro drivers chuẩn hóa</h2>
      <p className="muted">USD/VND · diesel · ENSO (ONI) · mưa 90 ngày · cân bằng nước. Dùng z-score để tránh biểu đồ hai trục.</p>
      <EChart option={option} height={540} />
    </section>
  );
}

function ReliabilityPanel({ horizon }: { horizon: number }) {
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setPayload(null);
    apiGet(`/reliability?horizon=${horizon}`).then(setPayload).catch((e: Error) => setError(e.message));
  }, [horizon]);
  if (error) return <ErrorBox message={error} />;
  if (!payload) return <Loading />;
  const deployed = payload.models.filter((row: any) => row.deployed);
  return (
    <>
      <div className="metric-grid three">
        <Metric label="Horizon" value={`h=${horizon}`} note={`${payload.out_of_sample_points} điểm OOS`} />
        <Metric label="Model deploy" value={String(deployed.length)} note="Có MAE/RMSE/DA trong registry" />
        <Metric label="Kết luận MCS" value="Không thắng RW" note="Stationary block bootstrap" />
      </div>
      <section className="panel table-panel">
        <h2>Model reliability</h2>
        <div className="table-scroll">
          <table>
            <thead><tr><th>Model</th><th>Deploy</th><th>MSE nghiên cứu</th><th>MAE deploy</th><th>RMSE deploy</th><th>DA</th></tr></thead>
            <tbody>{payload.models.map((row: any) => (
              <tr key={row.model}>
                <td>{row.model}</td><td>{row.deployed ? "Có" : "Offline"}</td>
                <td>{row.mse ?? "—"}</td><td>{row.mae ?? "—"}</td>
                <td>{row.rmse_deployed ?? "—"}</td><td>{row.directional_accuracy == null ? "—" : `${row.directional_accuracy}%`}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <div className="mcs-box"><b>{payload.mcs.conclusion}</b><span>{payload.mcs.provenance}</span></div>
      </section>
    </>
  );
}

export default function App() {
  const [active, setActive] = useState<Tab>("forecast");
  const [horizon, setHorizon] = useState(5);
  const [dark, setDark] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    apiGet<Meta>("/meta").then(setMeta).catch((reason: Error) => setApiError(reason.message));
  }, []);

  return (
    <div className={dark ? "app dark" : "app"}>
      <header className="hero">
        <div><span className="eyebrow">DAP391m · Group 6</span><h1>☕ CofPred</h1>
          <p>Coffee Price Forecasting and Decision Support Dashboard</p></div>
        <button className="theme-button" onClick={() => setDark((value) => !value)}>{dark ? "☀ Sáng" : "◐ Tối"}</button>
      </header>
      <main>
        {apiError && <ErrorBox message={`${apiError}. Hãy chạy FastAPI tại cổng 8000.`} />}
        <section className="study-strip">
          <div><b>{meta?.series_rows ?? "—"}</b><span>quan sát chuỗi</span></div>
          <div><b>{meta?.research_models ?? 22}</b><span>model nghiên cứu</span></div>
          <div><b>1 / 5 / 21 / 63</b><span>horizon nghiên cứu</span></div>
          <div><b>RandomWalk</b><span>benchmark</span></div>
        </section>
        <div className="toolbar">
          <nav>{TABS.map((tab) => <button key={tab.id} className={active === tab.id ? "active" : ""} onClick={() => setActive(tab.id)}>{tab.label}</button>)}</nav>
          {(["forecast", "comparison", "reliability"] as Tab[]).includes(active) && (
            <label>Horizon<select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
              {[1, 5, 21, 63].map((value) => <option key={value} value={value}>h={value}</option>)}
            </select></label>
          )}
        </div>
        <div className="content">
          {active === "forecast" && <ForecastPanel horizon={horizon} />}
          {active === "signal" && <SignalPanel />}
          {active === "comparison" && <ComparisonPanel horizon={horizon} />}
          {active === "basis" && <BasisPanel />}
          {active === "drivers" && <DriversPanel />}
          {active === "reliability" && <ReliabilityPanel horizon={horizon} />}
        </div>
      </main>
      <footer>Complex models do not beat the RandomWalk · Decision support, not investment advice.</footer>
    </div>
  );
}
