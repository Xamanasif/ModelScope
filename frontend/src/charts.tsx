import { ReactNode, useRef, useState } from "react";
import { fmt } from "./api";

export type Series = { key: string; label: string; color: string; points: [number, number][]; markers?: boolean; whiskers?: [number, number][] };

type Tip = { x: number; y: number; title: string; rows: { color: string; label: string; value: string }[] } | null;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x, top: tip.y }} role="presentation">
      <div className="t-title">{tip.title}</div>
      {tip.rows.map((r) => (
        <div className="t-row" key={r.label}>
          <span className="key" style={{ display: "inline-block", width: 12, height: 3, borderRadius: 2, background: r.color }} />
          <strong>{r.value}</strong>
          <span className="muted">{r.label}</span>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="legend" aria-hidden="true">
      {items.map((i) => (
        <span key={i.label}>
          <span className="key" style={{ background: i.dashed ? "transparent" : i.color, borderTop: i.dashed ? `2px dashed ${i.color}` : undefined, height: i.dashed ? 0 : 3 }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

const ticks = (lo: number, hi: number, n = 5) => Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);

/** Line/point chart on one pair of axes (never dual-axis). */
export function LineChart({
  series, xLabel, yLabel, xDomain = [0, 1], yDomain = [0, 1], diagonal, title, height = 300, xFmt = (v: number) => v.toFixed(2), tipTitle,
}: {
  series: Series[]; xLabel: string; yLabel: string; xDomain?: [number, number]; yDomain?: [number, number]; diagonal?: string; title: string; height?: number;
  xFmt?: (v: number) => string; tipTitle?: (x: number) => string;
}) {
  const W = 520, H = height, m = { l: 48, r: 14, t: 10, b: 40 };
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip>(null);
  const sx = (v: number) => m.l + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (W - m.l - m.r);
  const sy = (v: number) => H - m.b - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (H - m.t - m.b);
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - svg.left) / svg.width) * W;
    const xv = xDomain[0] + ((px - m.l) / (W - m.l - m.r)) * (xDomain[1] - xDomain[0]);
    if (xv < xDomain[0] || xv > xDomain[1]) return setTip(null);
    const rows = series.map((s) => {
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      return { color: s.color, label: s.label, value: `${xFmt(best[0])} → ${best[1].toFixed(3)}` };
    });
    const box = ref.current!.getBoundingClientRect();
    setTip({ x: Math.min(e.clientX - box.left + 12, box.width - 200), y: e.clientY - box.top + 12, title: tipTitle ? tipTitle(xv) : `${xLabel} ≈ ${xFmt(xv)}`, rows });
  };
  return (
    <div className="chart" ref={ref}>
      <Legend items={[...series.map((s) => ({ label: s.label, color: s.color })), ...(diagonal ? [{ label: diagonal, color: "var(--axis)", dashed: true }] : [])]} />
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
        {ticks(yDomain[0], yDomain[1]).map((t) => (
          <g key={`y${t}`}>
            <line className="gridline" x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        {ticks(xDomain[0], xDomain[1]).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - m.b + 16} textAnchor="middle">{xFmt(t)}</text>
        ))}
        <line className="axis" x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} />
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
        <text transform={`translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {diagonal && <line x1={sx(Math.max(xDomain[0], yDomain[0]))} y1={sy(Math.max(xDomain[0], yDomain[0]))} x2={sx(Math.min(xDomain[1], yDomain[1]))} y2={sy(Math.min(xDomain[1], yDomain[1]))} stroke="var(--axis)" strokeDasharray="4 4" strokeWidth={1.5} />}
        {series.map((s) => (
          <g key={s.key}>
            {s.whiskers?.map((w, i) => (
              <line key={i} x1={sx(s.points[i][0])} x2={sx(s.points[i][0])} y1={sy(w[0])} y2={sy(w[1])} stroke={s.color} strokeWidth={1.5} opacity={0.55} />
            ))}
            <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" points={s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />
            {s.markers && s.points.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />)}
          </g>
        ))}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

export type IntervalRow = { label: string; group?: string; values: { key: string; label: string; color: string; point: number | null; lo: number | null; hi: number | null; note?: string }[] };

/** Dot + interval ("forest") chart; one row per label, one dot per model. */
export function IntervalChart({ rows, domain, reference, referenceLabel, title, valueLabel }: { rows: IntervalRow[]; domain: [number, number]; reference?: number; referenceLabel?: string; title: string; valueLabel: string }) {
  const rowH = 30, W = 620, m = { l: 210, r: 20, t: 8, b: 32 };
  const per = rows[0]?.values.length ?? 1;
  const H = m.t + m.b + rows.length * rowH;
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip>(null);
  const sx = (v: number) => m.l + ((Math.min(Math.max(v, domain[0]), domain[1]) - domain[0]) / (domain[1] - domain[0])) * (W - m.l - m.r);
  const show = (e: React.PointerEvent | React.FocusEvent, r: IntervalRow) => {
    const box = ref.current!.getBoundingClientRect();
    const target = (e.target as Element).getBoundingClientRect();
    setTip({
      x: Math.min(target.left - box.left + 20, box.width - 220), y: target.top - box.top + 18, title: r.label,
      rows: r.values.map((v) => ({ color: v.color, label: v.label, value: v.point === null ? v.note ?? "unavailable" : `${fmt(v.point, 3)} [${fmt(v.lo, 3)}, ${fmt(v.hi, 3)}]` })),
    });
  };
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} onPointerLeave={() => setTip(null)}>
        {ticks(domain[0], domain[1], 4).map((t) => (
          <g key={t}>
            <line className="gridline" x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} />
            <text x={sx(t)} y={H - m.b + 15} textAnchor="middle">{t.toFixed(2)}</text>
          </g>
        ))}
        <text x={(m.l + W) / 2} y={H - 4} textAnchor="middle">{valueLabel}</text>
        {reference !== undefined && (
          <g>
            <line x1={sx(reference)} x2={sx(reference)} y1={m.t} y2={H - m.b} stroke="var(--ink-2)" strokeDasharray="3 3" />
            {referenceLabel && <text x={sx(reference) + 4} y={m.t + 10}>{referenceLabel}</text>}
          </g>
        )}
        {rows.map((r, i) => {
          const y0 = m.t + i * rowH;
          return (
            <g key={r.label} tabIndex={0} onPointerMove={(e) => show(e, r)} onFocus={(e) => show(e, r)} onBlur={() => setTip(null)}>
              <rect x={0} y={y0} width={W} height={rowH} fill="transparent" />
              <text x={m.l - 10} y={y0 + rowH / 2 + 4} textAnchor="end" style={{ fill: "var(--ink-2)" }}>{r.label.length > 34 ? r.label.slice(0, 33) + "…" : r.label}</text>
              {r.values.map((v, j) => {
                const y = y0 + rowH / 2 + (j - (per - 1) / 2) * 8;
                if (v.point === null) return <text key={v.key} x={m.l + 4} y={y + 4} className="unavailable">{j === 0 ? v.note ?? "unavailable" : ""}</text>;
                return (
                  <g key={v.key}>
                    {v.lo !== null && v.hi !== null && <line x1={sx(v.lo)} x2={sx(v.hi)} y1={y} y2={y} stroke={v.color} strokeWidth={2} strokeLinecap="round" />}
                    <circle cx={sx(v.point)} cy={y} r={4.5} fill={v.color} stroke="var(--surface)" strokeWidth={2} />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

export type BarRow = { label: string; values: { key: string; label: string; color: string; value: number | null }[] };

/** Horizontal bars from a zero baseline (supports negative values). */
export function BarChart({ rows, title, valueLabel, digits = 3, domain }: { rows: BarRow[]; title: string; valueLabel: string; digits?: number; domain?: [number, number] }) {
  const per = rows[0]?.values.length ?? 1;
  const barH = 9, gap = 2, rowH = per * (barH + gap) + 12, W = 620, m = { l: 210, r: 60, t: 6, b: 30 };
  const H = m.t + m.b + rows.length * rowH;
  const all = rows.flatMap((r) => r.values.map((v) => v.value ?? 0));
  const lo = domain ? domain[0] : Math.min(0, ...all), hi = domain ? domain[1] : Math.max(0, ...all);
  const span = hi - lo || 1;
  const sx = (v: number) => m.l + ((v - lo) / span) * (W - m.l - m.r);
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<Tip>(null);
  const show = (e: React.PointerEvent | React.FocusEvent, r: BarRow) => {
    const box = ref.current!.getBoundingClientRect();
    const t = (e.target as Element).getBoundingClientRect();
    setTip({ x: Math.min(t.left - box.left + 20, box.width - 220), y: t.top - box.top + 14, title: r.label, rows: r.values.map((v) => ({ color: v.color, label: v.label, value: fmt(v.value, digits) })) });
  };
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} onPointerLeave={() => setTip(null)}>
        {ticks(lo, hi, 4).map((t) => (
          <g key={t}>
            <line className="gridline" x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} />
            <text x={sx(t)} y={H - m.b + 15} textAnchor="middle">{t.toFixed(digits > 2 ? 2 : digits)}</text>
          </g>
        ))}
        <line className="axis" x1={sx(0)} x2={sx(0)} y1={m.t} y2={H - m.b} />
        <text x={(m.l + W) / 2} y={H - 3} textAnchor="middle">{valueLabel}</text>
        {rows.map((r, i) => {
          const y0 = m.t + i * rowH + 6;
          return (
            <g key={r.label} tabIndex={0} onPointerMove={(e) => show(e, r)} onFocus={(e) => show(e, r)} onBlur={() => setTip(null)}>
              <rect x={0} y={y0 - 6} width={W} height={rowH} fill="transparent" />
              <text x={m.l - 10} y={y0 + (per * (barH + gap)) / 2 + 3} textAnchor="end" style={{ fill: "var(--ink-2)" }}>{r.label.length > 34 ? r.label.slice(0, 33) + "…" : r.label}</text>
              {r.values.map((v, j) => {
                const y = y0 + j * (barH + gap);
                const val = v.value ?? 0;
                const x1 = sx(Math.min(0, val)), x2 = sx(Math.max(0, val));
                return <rect key={v.key} x={x1} y={y} width={Math.max(1, x2 - x1)} height={barH} rx={3} fill={v.color} />;
              })}
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

export function DataTable({ summary = "Show data table", children }: { summary?: string; children: ReactNode }) {
  return (
    <details style={{ marginTop: 8 }}>
      <summary>{summary}</summary>
      <div className="table-wrap" style={{ marginTop: 6 }}>{children}</div>
    </details>
  );
}
