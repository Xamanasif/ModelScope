import { useMemo, useState } from "react";
import { api, ci, fmt, Json, pct, ROLE_COLOR, ROLE_LABEL, ROLE_SHORT, signed } from "./api";
import { DataTable, IntervalChart, LineChart } from "./charts";
import { useStudy } from "./ctx";
import { Async, Badge, CheckBadge, Empty, ErrorBox, NoStudy, Seg, Stat, StateBadge, useApi } from "./ui";

const ROLES = ["baseline", "challenger"] as const;
const METRICS: [string, string, string][] = [
  ["roc_auc", "ROC-AUC", "ranking; higher is better"],
  ["average_precision", "Average precision", "step-wise AP, not trapezoidal PR-AUC; higher is better"],
  ["log_loss", "Log loss", "proper scoring rule; lower is better"],
  ["brier", "Brier score", "proper scoring rule; lower is better"],
];

// ------------------------------------------------------------------ Overview
export function OverviewPage() {
  const { status, studyId, study } = useStudy();
  const dev = useApi(studyId ? `/api/studies/${studyId}/evidence/development` : null);
  const fin = useApi(studyId && study?.final_evaluated ? `/api/studies/${studyId}/evidence/final` : null);
  const rev = useApi(studyId ? `/api/studies/${studyId}/review` : null);
  const man = status?.source_manifest;
  const steps = [
    { label: "Official dataset acquired and validated", done: !!status?.dataset_present, detail: man ? `SHA-256 ${man.sha256.slice(0, 12)}… · ${man.rows.toLocaleString()} rows · accessed ${man.accessed_utc.slice(0, 10)}` : "not downloaded" },
    { label: "Development study (train / tuning / calibration only)", done: !!study, detail: study ? `${study.study_id} · ${study.status}` : "not run" },
    { label: "Champion selection frozen", done: !!study?.champion, detail: study?.champion ? `${ROLE_LABEL[study.champion]}` : "—" },
    { label: "Final-holdout audit (one shot)", done: !!study?.final_evaluated, detail: study?.final_evaluated ? "evaluated once" : "sealed" },
    { label: "Human review decision", done: ["accepted_for_research", "rejected", "changes_requested"].includes(study?.review_state), detail: study?.review_state?.replace(/_/g, " ") ?? "—" },
  ];
  return (
    <div className="stack">
      <div className="callout">
        <strong>Central question:</strong> does better ranking (AUC) also give reliable probabilities and stable performance across customer
        segments? Everything here is a local research prototype. No production use, regulatory compliance or bank approval is claimed.
      </div>
      <div className="grid four">
        <Stat label="Customers" value={man ? man.rows.toLocaleString() : "—"} note="UCI 350, Taiwan 2005, cross-sectional" />
        <Stat label="Default prevalence" value={dev.data ? pct(dev.data.prevalence.train, 2) : "—"} note="train partition; positive class = default (Y=1)" />
        <Stat label="Champion (frozen rule)" value={study?.champion ? ROLE_SHORT[study.champion] : "—"} note="development evidence only" />
        <Stat label="Review state" value={study ? <StateBadge state={study.review_state} /> : "—"} note="decided by a human, never automatically" />
      </div>
      <div className="grid two">
        <div className="card">
          <h2>Experiment status</h2>
          <ol className="timeline">
            {steps.map((s) => (
              <li key={s.label}>
                <Badge tone={s.done ? "good" : "neutral"}>{s.done ? "done" : "open"}</Badge> <strong>{s.label}</strong>
                <div className="muted small">{s.detail}</div>
              </li>
            ))}
          </ol>
        </div>
        <div className="card">
          <h2>Major findings and flags</h2>
          {!studyId ? <p className="muted">No study yet.</p> : (
            <Async state={rev}>
              {(r) => {
                const open = r.findings.filter((f: Json) => ["open", "in_remediation"].includes(f.status));
                const flags = r.checks.filter((c: Json) => c.status === "flag");
                return (
                  <>
                    <p className="small muted">{open.length} open finding(s) · {flags.length} deterministic flag(s). Flags are prompts for a reviewer, not decisions.</p>
                    <ul className="small">
                      {open.slice(0, 5).map((f: Json) => <li key={f.finding_id}><Badge tone={f.severity === "critical" || f.severity === "high" ? "crit" : "warn"}>{f.severity}</Badge> {f.title}</li>)}
                      {flags.slice(0, 6).map((c: Json) => <li key={c.check_id}><CheckBadge status="flag" /> {c.message}</li>)}
                    </ul>
                    {open.length + flags.length === 0 && <p className="small">No open findings or flags.</p>}
                  </>
                );
              }}
            </Async>
          )}
        </div>
      </div>
      {!studyId ? <NoStudy /> : (
        <div className="card">
          <h2>Development vs final results (calibrated probabilities)</h2>
          <Async state={dev}>
            {(d) => (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Model</th><th>Partition</th>{METRICS.map(([k, l, h]) => <th key={k} className="num" title={h}>{l}</th>)}</tr></thead>
                  <tbody>
                    {ROLES.flatMap((r) => [["tuning", d.evaluation[r].calibrated.metrics], ["final", fin.data ? fin.data.evaluation[r].metrics : null]].map(([p, m]: Json) => (
                      <tr key={r + p}>
                        <td className="nowrap">{p === "tuning" ? <>{ROLE_SHORT[r]}{study?.champion === r && <> <Badge tone="plain">champion</Badge></>}</> : ""}</td>
                        <td>{p === "tuning" ? "development (tuning)" : "final holdout audit"}</td>
                        {METRICS.map(([k]) => <td key={k} className="num">{m ? fmt(m[k]) : <span className="unavailable">sealed</span>}</td>)}
                      </tr>
                    )))}
                  </tbody>
                </table>
                <p className="small muted" style={{ marginTop: 8 }}>
                  Tuning n = {d.partition_sizes.tuning.toLocaleString()}; final n = {d.partition_sizes.final.toLocaleString()}. The final holdout is used once, after selection is frozen, as an audit.
                </p>
              </div>
            )}
          </Async>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ Data quality
export function DataPage() {
  const q = useApi("/api/data/quality");
  const [upload, setUpload] = useState<{ state: "idle" | "validating" | "done" | "error"; result?: Json; error?: string }>({ state: "idle" });
  const onFile = (f: File | undefined) => {
    if (!f) return;
    setUpload({ state: "validating" });
    api.postCsv("/api/data/validate-upload", f).then((r) => setUpload({ state: "done", result: r })).catch((e) => setUpload({ state: "error", error: e.message }));
  };
  return (
    <Async state={q}>
      {(d) => !d.available ? <Empty title="No dataset available">{d.reason}<br />Run <code>python scripts/download_data.py</code> (or the manual route in README).</Empty> : (
        <div className="stack">
          <div className="grid two">
            <div className="card">
              <h2>Provenance</h2>
              {d.source_manifest ? (
                <table><tbody>
                  {[["Source", d.source_manifest.landing_page], ["Route", d.source_manifest.acquisition_route], ["Accessed (UTC)", d.source_manifest.accessed_utc], ["SHA-256", d.source_manifest.sha256], ["Rows", d.source_manifest.rows], ["Citation", d.source_manifest.citation], ["License", d.source_manifest.license], ["Target", d.source_manifest.target_semantics], ["Metadata mapping re-checked", String(d.source_manifest.metadata_mapping_checked)]].map(([k, v]) => (
                    <tr key={k as string}><th style={{ width: 160 }}>{k}</th><td className={k === "SHA-256" ? "mono" : ""} style={{ wordBreak: "break-all" }}>{String(v)}</td></tr>
                  ))}
                </tbody></table>
              ) : <p className="muted">No source manifest found.</p>}
            </div>
            <div className="card">
              <h2>Split isolation</h2>
              <p className="small">Customer-level stratified split, seed {d.split.seed}. {d.temporal_note}</p>
              <table>
                <thead><tr><th>Partition</th><th className="num">Customers</th><th className="num">Default rate</th><th>Used for</th></tr></thead>
                <tbody>
                  {[["train", "fitting preprocessing and models; CV hyperparameter search"], ["tuning", "calibration choice, champion rule, slices, stress, SHAP"], ["calibration", "fitting the sigmoid map only"], ["final", "one-shot audit after selection is frozen"]].map(([k, u]) => (
                    <tr key={k}><td>{k}</td><td className="num">{d.split.counts[k].toLocaleString()}</td><td className="num">{pct(d.split.prevalence[k], 2)}</td><td className="small">{u}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="small" style={{ marginTop: 8 }}>
                Pairwise overlaps: {Object.entries(d.split.pairwise_overlap).map(([k, v]) => `${k} ${v}`).join(", ")} ·
                union equals dataset: <strong>{String(d.split.union_equals_dataset)}</strong>
              </p>
              <p className="small mono" style={{ wordBreak: "break-all" }}>manifest {d.split.manifest_sha256}{d.split.expected_manifest_sha256 && (d.split.expected_manifest_sha256 === d.split.manifest_sha256 ? " ✓ matches pre-registered hash" : " ✗ differs from pre-registered hash")}</p>
            </div>
          </div>
          <div className="card">
            <h2>Schema and domain checks</h2>
            <p className="small muted">{d.quality.summary.pass} pass · {d.quality.summary.info} info · {d.quality.summary.warn} warn · {d.quality.summary.fail} fail. Warnings mark undocumented codes that are kept, not reinterpreted.</p>
            <div className="table-wrap"><table>
              <thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
              <tbody>{d.quality.checks.map((c: Json) => <tr key={c.name}><td className="mono">{c.name}</td><td><CheckBadge status={c.status} /></td><td className="small">{c.detail}</td></tr>)}</tbody>
            </table></div>
          </div>
          <div className="grid two">
            <div className="card">
              <h2>Code semantics (from the source)</h2>
              <table><tbody>{Object.entries(d.code_tables).map(([k, v]) => <tr key={k}><th style={{ width: 120 }}>{k}</th><td className="small">{String(v)}</td></tr>)}</tbody></table>
              <p className="small muted" style={{ marginTop: 8 }}>Demographic columns are excluded from predictors by default and used for audit slices only (policy <code>features.demographic_predictors = false</code>).</p>
            </div>
            <div className="card">
              <h2>Validate a CSV file</h2>
              <p className="small">Checks a file against the UCI 350 schema. Nothing is stored and nothing is trained.</p>
              <input type="file" accept=".csv,text/csv" aria-label="CSV file to validate" onChange={(e) => onFile(e.target.files?.[0])} />
              <div aria-live="polite" style={{ marginTop: 10 }}>
                {upload.state === "validating" && <p className="small">Validating…</p>}
                {upload.state === "error" && <ErrorBox message={upload.error!} />}
                {upload.state === "done" && (
                  <>
                    <p>{upload.result.accepted ? <Badge tone="good">valid: passes all hard checks</Badge> : <Badge tone="crit">rejected: hard check failed</Badge>} <span className="small muted">({upload.result.report.rows} rows, not stored)</span></p>
                    <ul className="small">{upload.result.report.checks.filter((c: Json) => c.status !== "pass").map((c: Json) => <li key={c.name}><CheckBadge status={c.status} /> {c.name}: {c.detail}</li>)}</ul>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </Async>
  );
}

// ------------------------------------------------------------------ Comparison
export function ComparisonPage() {
  const { studyId, study } = useStudy();
  const [part, setPart] = useState<"tuning" | "final">("tuning");
  const dev = useApi(studyId ? `/api/studies/${studyId}/evidence/development` : null);
  const fin = useApi(studyId && study?.final_evaluated ? `/api/studies/${studyId}/evidence/final` : null);
  const sel = useApi(studyId ? `/api/studies/${studyId}/evidence/selection` : null);
  if (!studyId) return <NoStudy />;
  return (
    <Async state={dev}>
      {(d) => {
        const finalReady = !!fin.data;
        const evalFor = (r: string) => (part === "final" && finalReady ? fin.data.evaluation[r] : d.evaluation[r].calibrated);
        const boot = part === "final" && finalReady ? fin.data.paired_bootstrap_final : d.paired_bootstrap_tuning;
        const n = part === "final" && finalReady ? fin.data.n : d.partition_sizes.tuning;
        const prev = part === "final" && finalReady ? fin.data.prevalence : d.prevalence.tuning;
        return (
          <div className="stack">
            <div className="row">
              <Seg label="Evaluation partition" value={part} onChange={setPart} options={[{ value: "tuning", label: "Development (tuning)" }, { value: "final", label: "Final holdout audit" }]} />
              <span className="small muted">n = {n.toLocaleString()} · default prevalence {pct(prev, 2)} · threshold {evalFor("baseline").metrics.threshold} (illustrative)</span>
            </div>
            {part === "final" && !finalReady ? <Empty title="Final holdout still sealed">Run the one-shot audit from Reports &amp; runs once you accept that the selection is frozen.</Empty> : (
              <>
                <div className="callout">
                  <strong>Three different questions.</strong> ROC-AUC asks whether defaulters rank above non-defaulters. Log loss, Brier and the
                  calibration curve ask whether a stated 30% really means about 30% of such customers default. The threshold table asks what happens
                  if everyone above a cut-off is flagged. A model can win one and lose another.
                </div>
                <div className="card">
                  <h2>Metrics (calibrated probabilities)</h2>
                  <div className="table-wrap"><table>
                    <thead><tr><th>Model</th>{METRICS.map(([k, l, h]) => <th key={k} className="num" title={h}>{l}</th>)}<th className="num">ECE (10 bins)</th><th className="num">Obs − pred</th><th className="num">Calib. slope</th></tr></thead>
                    <tbody>{ROLES.map((r) => { const e = evalFor(r); return (
                      <tr key={r}><td><span className="key" style={{ display: "inline-block", width: 12, height: 3, background: ROLE_COLOR[r], marginRight: 6, verticalAlign: "middle" }} />{ROLE_LABEL[r]}</td>
                        {METRICS.map(([k]) => <td key={k} className="num">{fmt(e.metrics[k])}</td>)}
                        <td className="num">{fmt(e.calibration.ece_quantile_10)}</td><td className="num">{signed(e.calibration.calibration_in_the_large)}</td><td className="num">{fmt(e.calibration.slope, 3)}</td></tr>); })}
                    </tbody>
                  </table></div>
                  <p className="small muted" style={{ marginTop: 6 }}>ECE is a binning-dependent diagnostic, not a proper scoring rule. A calibration slope below 1 means predictions are too extreme.</p>
                </div>
                <div className="grid two">
                  <div className="card"><h3>ROC curve</h3>
                    <LineChart title="ROC curves" xLabel="False positive rate" yLabel="True positive rate" diagonal="chance"
                      series={ROLES.map((r) => ({ key: r, label: ROLE_SHORT[r], color: ROLE_COLOR[r], points: evalFor(r).curves.roc }))} />
                  </div>
                  <div className="card"><h3>Precision–recall curve</h3>
                    <LineChart title="Precision-recall curves" xLabel="Recall" yLabel="Precision"
                      series={ROLES.map((r) => ({ key: r, label: ROLE_SHORT[r], color: ROLE_COLOR[r], points: evalFor(r).curves.pr }))} />
                    <p className="small muted">A random ranking would give precision ≈ prevalence ({pct(prev, 1)}).</p>
                  </div>
                </div>
                <div className="card">
                  <h3>Calibration (reliability) curve, 10 equal-count bins</h3>
                  <CalibrationView evalFor={evalFor} />
                </div>
                <ThresholdView evalFor={evalFor} />
                <div className="card">
                  <h2>Paired bootstrap: challenger − baseline</h2>
                  <p className="small">{boot.method}; {boot.n_used} resamples used, {boot.n_failed} failed (one class only) and excluded. Intervals are conditional on the fitted models and this evaluation sample; they do not cover training-sample uncertainty.</p>
                  <div className="table-wrap"><table>
                    <thead><tr><th>Metric</th><th className="num">Baseline</th><th className="num">Challenger</th><th className="num">Difference</th><th className="num">95% CI</th><th>Reading</th></tr></thead>
                    <tbody>{METRICS.map(([k, l]) => { const v = boot.metrics[k]; const excl = v.difference_ci[0] > 0 || v.difference_ci[1] < 0; const better = v.direction === "higher_is_better" ? v.difference_challenger_minus_baseline > 0 : v.difference_challenger_minus_baseline < 0; return (
                      <tr key={k}><td>{l}</td><td className="num">{fmt(v.baseline)}</td><td className="num">{fmt(v.challenger)}</td><td className="num">{signed(v.difference_challenger_minus_baseline)}</td><td className="num">{ci(v.difference_ci)}</td>
                        <td className="small">{excl ? `interval excludes 0: challenger ${better ? "better" : "worse"}` : "interval includes 0: no clear difference"}</td></tr>); })}
                    </tbody>
                  </table></div>
                  <IntervalChart title="Difference in ROC-AUC with 95% interval" valueLabel="ROC-AUC difference (challenger − baseline)" reference={0} referenceLabel="no difference"
                    domain={[Math.min(-0.02, boot.metrics.roc_auc.difference_ci[0] - 0.01), Math.max(0.02, boot.metrics.roc_auc.difference_ci[1] + 0.01)]}
                    rows={[{ label: "ROC-AUC difference", values: [{ key: "d", label: "challenger − baseline", color: "var(--ink-2)", point: boot.metrics.roc_auc.difference_challenger_minus_baseline, lo: boot.metrics.roc_auc.difference_ci[0], hi: boot.metrics.roc_auc.difference_ci[1] }] }]} />
                </div>
              </>
            )}
            <Async state={sel}>
              {(s) => (
                <div className="card">
                  <h2>Frozen selection record</h2>
                  <p className="small">{s.rule}</p>
                  <p>Log-loss difference CI on tuning: <strong>{ci(s.log_loss_difference_ci)}</strong> → champion <strong>{ROLE_LABEL[s.champion]}</strong>. Frozen {s.frozen_utc}.</p>
                  <p className="small muted">Calibration: {ROLES.map((r) => `${ROLE_SHORT[r]} ${s.calibration[r].method} (a=${fmt(s.calibration[r].a, 3)}, b=${fmt(s.calibration[r].b, 3)})`).join(" · ")}</p>
                </div>
              )}
            </Async>
            <ModelDetails d={d} />
            <CostScenario studyId={studyId} finalReady={finalReady} />
          </div>
        );
      }}
    </Async>
  );
}

function CalibrationView({ evalFor }: { evalFor: (r: string) => Json }) {
  const max = Math.min(1, Math.max(...ROLES.flatMap((r) => evalFor(r).calibration.table.map((b: Json) => Math.max(b.mean_predicted, b.observed_ci_high)))) * 1.08);
  return (
    <>
      <div className="chart narrow"><LineChart title="Calibration curves" height={380} xLabel="Mean predicted default probability (bin)" yLabel="Observed default rate" diagonal="perfect calibration" xDomain={[0, max]} yDomain={[0, max]}
        series={ROLES.map((r) => ({ key: r, label: ROLE_SHORT[r], color: ROLE_COLOR[r], markers: true,
          points: evalFor(r).calibration.table.map((b: Json) => [b.mean_predicted, b.observed_rate] as [number, number]),
          whiskers: evalFor(r).calibration.table.map((b: Json) => [b.observed_ci_low, b.observed_ci_high] as [number, number]) }))} /></div>
      <p className="small muted">Vertical lines: Wilson 95% intervals for each bin’s observed rate (binomial noise only).</p>
      <DataTable>
        <table><thead><tr><th>Model</th><th className="num">Bin</th><th className="num">n</th><th className="num">Mean predicted</th><th className="num">Observed</th><th className="num">Observed 95% CI</th></tr></thead>
          <tbody>{ROLES.flatMap((r) => evalFor(r).calibration.table.map((b: Json, i: number) => (
            <tr key={r + i}><td>{ROLE_SHORT[r]}</td><td className="num">{i + 1}</td><td className="num">{b.n}</td><td className="num">{fmt(b.mean_predicted, 3)}</td><td className="num">{fmt(b.observed_rate, 3)}</td><td className="num">{ci([b.observed_ci_low, b.observed_ci_high], 3)}</td></tr>)))}</tbody></table>
      </DataTable>
    </>
  );
}

function ThresholdView({ evalFor }: { evalFor: (r: string) => Json }) {
  const grid: number[] = evalFor("baseline").threshold_grid.map((t: Json) => t.threshold);
  const [t, setT] = useState<number>(grid.includes(0.5) ? 0.5 : grid[0]);
  return (
    <div className="card">
      <h2>Threshold trade-offs</h2>
      <div className="row">
        <label className="field" style={{ width: 200 }}><span>Threshold on calibrated probability</span>
          <select value={t} onChange={(e) => setT(Number(e.target.value))}>{grid.map((g) => <option key={g} value={g}>{g.toFixed(2)}</option>)}</select>
        </label>
        <span className="small muted">0.5 is the pre-registered illustrative threshold, not a lending policy.</span>
      </div>
      <div className="table-wrap" style={{ marginTop: 10 }}><table>
        <thead><tr><th>Model</th><th className="num">Flag rate</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">Specificity</th><th className="num">TP</th><th className="num">FP</th><th className="num">FN</th><th className="num">TN</th></tr></thead>
        <tbody>{ROLES.map((r) => { const row = evalFor(r).threshold_grid.find((g: Json) => g.threshold === t); const cm = row.confusion_matrix; return (
          <tr key={r}><td>{ROLE_SHORT[r]}</td><td className="num">{pct(row.flag_rate)}</td><td className="num">{fmt(row.precision, 3)}</td><td className="num">{fmt(row.recall, 3)}</td><td className="num">{fmt(row.specificity, 3)}</td><td className="num">{cm.tp}</td><td className="num">{cm.fp}</td><td className="num">{cm.fn}</td><td className="num">{cm.tn}</td></tr>); })}</tbody>
      </table></div>
    </div>
  );
}

function ModelDetails({ d }: { d: Json }) {
  return (
    <div className="card">
      <h2>Configurations, search and training time</h2>
      <div className="table-wrap"><table>
        <thead><tr><th>Model</th><th>Untuned</th><th>Selected (CV inside train)</th><th className="num">Untuned AUC / log loss</th><th className="num">Selected AUC / log loss</th><th className="num">Search s</th><th className="num">Refit s</th><th className="num">Configs</th></tr></thead>
        <tbody>{ROLES.map((r) => { const m = d.models[r]; const u = d.evaluation[r].untuned_uncalibrated_metrics; const s = d.evaluation[r].uncalibrated_metrics; return (
          <tr key={r}><td>{ROLE_SHORT[r]}<div className="small muted">{m.preprocessing_version}</div></td><td className="mono small">{JSON.stringify(m.untuned_params)}</td><td className="mono small">{JSON.stringify(m.selected_params)}</td>
            <td className="num">{fmt(u.roc_auc)} / {fmt(u.log_loss)}</td><td className="num">{fmt(s.roc_auc)} / {fmt(s.log_loss)}</td><td className="num">{m.search_seconds}</td><td className="num">{m.refit_seconds}</td><td className="num">{m.n_configurations}</td></tr>); })}</tbody>
      </table></div>
      <p className="small muted">Untuned/selected rows are uncalibrated, on the tuning partition. Calibration decision per model: {ROLES.map((r) => `${ROLE_SHORT[r]}: tuning log loss none ${fmt(d.models[r].calibration.tuning_log_loss.none)} vs sigmoid ${fmt(d.models[r].calibration.tuning_log_loss.sigmoid)} → ${d.models[r].calibration.method}`).join("; ")}.</p>
      <DataTable summary="Show every CV configuration">
        {ROLES.map((r) => (
          <table key={r} style={{ marginBottom: 12 }}><caption className="small" style={{ textAlign: "left" }}>{ROLE_LABEL[r]}</caption>
            <thead><tr><th>Params</th><th className="num">CV log loss</th><th className="num">sd</th><th className="num">CV AUC</th><th className="num">Fit s</th></tr></thead>
            <tbody>{d.models[r].search_results.map((x: Json, i: number) => <tr key={i}><td className="mono small">{JSON.stringify(x.params)}{x.is_untuned_default && " (untuned)"}</td><td className="num">{fmt(x.cv_log_loss_mean)}</td><td className="num">{fmt(x.cv_log_loss_sd)}</td><td className="num">{fmt(x.cv_roc_auc_mean)}</td><td className="num">{x.fit_seconds_total}</td></tr>)}</tbody></table>
        ))}
      </DataTable>
      <h3 style={{ marginTop: 14 }}>Repeated development resamples</h3>
      <p className="small">{d.repeated_cv.scheme}. AUC difference mean {signed(d.repeated_cv.summary.auc_difference.mean)} (sd {fmt(d.repeated_cv.summary.auc_difference.sd)}, range {fmt(d.repeated_cv.summary.auc_difference.min)} to {fmt(d.repeated_cv.summary.auc_difference.max)}); challenger had lower log loss in {d.repeated_cv.summary.folds_challenger_better_log_loss}/{d.repeated_cv.folds.length} folds. <span className="muted">{d.repeated_cv.caveat}</span></p>
    </div>
  );
}

function CostScenario({ studyId, finalReady }: { studyId: string; finalReady: boolean }) {
  const [fp, setFp] = useState("1");
  const [fn, setFn] = useState("5");
  const [part, setPart] = useState<"tuning" | "final">("tuning");
  const [res, setRes] = useState<{ loading?: boolean; data?: Json; error?: string }>({});
  const run = () => {
    setRes({ loading: true });
    api.get(`/api/studies/${studyId}/threshold-scenario?cost_fp=${encodeURIComponent(fp)}&cost_fn=${encodeURIComponent(fn)}&partition_name=${part}`).then((d) => setRes({ data: d })).catch((e) => setRes({ error: e.message }));
  };
  return (
    <div className="card">
      <h2>Decision-cost scenario <Badge tone="sim">user-supplied assumptions</Badge></h2>
      <p className="small">Enter hypothetical relative costs. If probabilities are calibrated, flagging when p &gt; c<sub>FP</sub>/(c<sub>FP</sub>+c<sub>FN</sub>) minimises expected cost. These are not observed banking losses, exposures or ECL/IFRS 9 figures.</p>
      <div className="row">
        <label className="field" style={{ width: 160 }}><span>Cost of a false positive</span><input type="number" min="0" step="any" value={fp} onChange={(e) => setFp(e.target.value)} /></label>
        <label className="field" style={{ width: 160 }}><span>Cost of a missed default</span><input type="number" min="0" step="any" value={fn} onChange={(e) => setFn(e.target.value)} /></label>
        <Seg label="Scenario partition" value={part} onChange={setPart} options={[{ value: "tuning", label: "Tuning" }, ...(finalReady ? [{ value: "final" as const, label: "Final" }] : [])]} />
        <button className="btn" onClick={run} disabled={res.loading}>{res.loading ? "Computing…" : "Compute"}</button>
      </div>
      {res.error && <div style={{ marginTop: 10 }}><ErrorBox message={res.error} /></div>}
      {res.data && (
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <p className="small">Threshold {fmt(res.data.threshold, 3)} on the {res.data.partition} partition.</p>
          <table><thead><tr><th>Model</th><th className="num">Flag rate</th><th className="num">Recall</th><th className="num">Precision</th><th className="num">Scenario cost / customer</th></tr></thead>
            <tbody>{ROLES.map((r) => { const m = res.data.models[r]; return <tr key={r}><td>{ROLE_SHORT[r]}</td><td className="num">{pct(m.flag_rate)}</td><td className="num">{fmt(m.recall, 3)}</td><td className="num">{fmt(m.precision, 3)}</td><td className="num">{fmt(m.scenario_cost_per_customer, 4)}</td></tr>; })}</tbody></table>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ Segments
export function SegmentsPage() {
  const { studyId, study } = useStudy();
  const [part, setPart] = useState<"tuning" | "final">("tuning");
  const name = part === "final" ? "slices_final" : "slices_tuning";
  const s = useApi(studyId && (part === "tuning" || study?.final_evaluated) ? `/api/studies/${studyId}/evidence/${name}` : null, [part]);
  const [family, setFamily] = useState("credit_limit_band");
  if (!studyId) return <NoStudy />;
  return (
    <div className="stack">
      <div className="row">
        <Seg label="Partition" value={part} onChange={setPart} options={[{ value: "tuning", label: "Development (tuning)" }, { value: "final", label: "Final holdout" }]} />
      </div>
      {part === "final" && !study?.final_evaluated ? <Empty title="Final holdout still sealed" /> : (
        <Async state={s}>
          {(d) => <SliceView d={d} family={family} setFamily={setFamily} />}
        </Async>
      )}
    </div>
  );
}

function SliceView({ d, family, setFamily }: { d: Json; family: string; setFamily: (f: string) => void }) {
  const families: string[] = useMemo(() => Array.from(new Set(d.rows.map((r: Json) => r.slice as string))), [d]);
  const rows = d.rows.filter((r: Json) => r.slice === family);
  const isDemo = rows[0]?.demographic;
  return (
    <>
      <div className="callout warn"><strong>Read with care.</strong> {d.note} {d.uncertainty} Small groups have wide intervals; differences between groups are not tests of discrimination and do not imply any action towards a group.</div>
      <div className="card">
        <div className="row">
          <label className="field" style={{ width: 260 }}><span>Slice family</span>
            <select value={family} onChange={(e) => setFamily(e.target.value)}>{families.map((f) => <option key={f} value={f}>{f.replace(/_/g, " ")}</option>)}</select>
          </label>
          {isDemo && <Badge tone="warn">demographic audit slice: not a predictor by default</Badge>}
        </div>
        <h3 style={{ marginTop: 12 }}>ROC-AUC by group with 95% bootstrap interval</h3>
        <IntervalChart title={`AUC by ${family}`} valueLabel="ROC-AUC within group" domain={[0.4, 1]}
          rows={rows.map((r: Json) => ({ label: `${r.group} (n=${r.n})`, values: ["baseline", "challenger"].map((m) => {
            const v = r.models[m];
            return { key: m, label: ROLE_SHORT[m], color: ROLE_COLOR[m], point: v?.roc_auc ?? null, lo: v?.roc_auc_ci?.[0] ?? null, hi: v?.roc_auc_ci?.[1] ?? null, note: v?.roc_auc_unavailable_reason ?? r.unavailable_reason ?? "unavailable" };
          }) }))} />
        <div className="legend"><span><span className="key" style={{ background: "var(--series-1)" }} />{ROLE_SHORT.baseline}</span><span><span className="key" style={{ background: "var(--series-2)" }} />{ROLE_SHORT.challenger}</span></div>
        <div className="table-wrap" style={{ marginTop: 10 }}><table>
          <thead><tr><th>Group</th><th className="num">n</th><th className="num">Defaults</th><th className="num">Prevalence</th>{["baseline", "challenger"].map((m) => <th key={m} className="num">{ROLE_SHORT[m]} AUC [CI]</th>)}{["baseline", "challenger"].map((m) => <th key={m + "g"} className="num">{ROLE_SHORT[m]} obs − pred [CI]</th>)}<th className="num">Recall / FPR @thr (LR · HGB)</th><th>Cautions</th></tr></thead>
          <tbody>{rows.map((r: Json) => (
            <tr key={r.group} className={r.warnings.length ? "dim" : ""}>
              <td className="nowrap">{r.group}</td><td className="num">{r.n}</td><td className="num">{r.n_default}</td><td className="num">{pct(r.prevalence)}</td>
              {["baseline", "challenger"].map((m) => <td key={m} className="num">{r.models[m]?.roc_auc === null || !r.models[m] ? <span className="unavailable">{r.models[m]?.roc_auc_unavailable_reason ?? r.unavailable_reason}</span> : <>{fmt(r.models[m].roc_auc, 3)} {ci(r.models[m].roc_auc_ci, 3)}</>}</td>)}
              {["baseline", "challenger"].map((m) => <td key={m + "g"} className="num">{r.models[m] ? <>{signed(r.models[m].observed_minus_predicted, 3)} {ci(r.models[m].observed_minus_predicted_ci, 3)}</> : "n/a"}</td>)}
              <td className="num">{["baseline", "challenger"].map((m) => r.models[m] ? `${fmt(r.models[m].recall, 2)}/${fmt(r.models[m].false_positive_rate, 2)}` : "n/a").join(" · ")}</td>
              <td className="small">{[...r.warnings, ...["baseline", "challenger"].map((m) => r.models[m]?.roc_auc_caution).filter(Boolean)].join("; ") || "—"}</td>
            </tr>))}</tbody>
        </table></div>
        {family === "credit_limit_band" && <p className="small muted">Band edges fitted on the train partition: {d.edges.LIMIT_BAL_quartile_edges.map((e: number) => e.toLocaleString()).join(", ")} NT$.</p>}
      </div>
    </>
  );
}
