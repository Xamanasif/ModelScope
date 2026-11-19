import { useEffect, useMemo, useState } from "react";
import { api, fmt, Json, pct, ROLE_COLOR, ROLE_LABEL, ROLE_SHORT, signed, storageGet, storageSet } from "./api";
import { BarChart, DataTable } from "./charts";
import { useStudy } from "./ctx";
import { Async, Badge, CheckBadge, Empty, ErrorBox, NoStudy, Seg, StateBadge, useApi } from "./ui";

const ROLES = ["baseline", "challenger"] as const;

function newRequestId() {
  try {
    return crypto.randomUUID();
  } catch {
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}

function useActor(): [string, (v: string) => void] {
  const [actor, setActor] = useState(storageGet("modelscope.actor") ?? "");
  return [actor, (v: string) => { setActor(v); storageSet("modelscope.actor", v); }];
}

// ------------------------------------------------------------------ Stress & shift
export function StressPage() {
  const { studyId } = useStudy();
  const st = useApi(studyId ? `/api/studies/${studyId}/evidence/stress` : null);
  const dr = useApi(studyId ? `/api/studies/${studyId}/evidence/drift` : null);
  const [metric, setMetric] = useState<"delta_roc_auc" | "delta_log_loss">("delta_roc_auc");
  if (!studyId) return <NoStudy />;
  return (
    <div className="stack">
      <Async state={st}>
        {(s) => (
          <div className="card">
            <h2>Stress suite <span className="small muted">({s.partition} partition, seeded)</span></h2>
            <div className="callout">{s.caveat}</div>
            <div className="row" style={{ marginBottom: 8 }}>
              <Seg label="Stress metric" value={metric} onChange={setMetric} options={[{ value: "delta_roc_auc", label: "Change in AUC" }, { value: "delta_log_loss", label: "Change in log loss" }]} />
              <span className="small muted">Relative to clean tuning data: AUC {ROLES.map((r) => `${ROLE_SHORT[r]} ${fmt(s.clean[r].roc_auc)}`).join(", ")}</span>
            </div>
            <BarChart title="Change in metric under each stress scenario" valueLabel={metric === "delta_roc_auc" ? "Δ ROC-AUC (negative = worse)" : "Δ log loss (positive = worse)"} digits={4}
              rows={s.scenarios.map((sc: Json) => ({ label: sc.id, values: ROLES.map((r) => ({ key: r, label: ROLE_SHORT[r], color: ROLE_COLOR[r], value: sc.models[r][metric] })) }))} />
            <div className="legend">{ROLES.map((r) => <span key={r}><span className="key" style={{ background: ROLE_COLOR[r] }} />{ROLE_SHORT[r]}</span>)}</div>
            <div className="table-wrap"><table>
              <thead><tr><th>Scenario</th><th className="num">n / prevalence</th>{ROLES.map((r) => <th key={r} className="num">{ROLE_SHORT[r]} AUC (Δ)</th>)}{ROLES.map((r) => <th key={r + "l"} className="num">{ROLE_SHORT[r]} log loss (Δ)</th>)}<th className="num">Mean pred LR · HGB</th></tr></thead>
              <tbody>{s.scenarios.map((sc: Json) => (
                <tr key={sc.id}><td className="mono small nowrap">{sc.id}</td>
                  <td className="num">{sc.n} / {pct(sc.prevalence)}</td>
                  {ROLES.map((r) => <td key={r} className="num">{fmt(sc.models[r].roc_auc)} ({signed(sc.models[r].delta_roc_auc)})</td>)}
                  {ROLES.map((r) => <td key={r + "l"} className="num">{fmt(sc.models[r].log_loss)} ({signed(sc.models[r].delta_log_loss)})</td>)}
                  <td className="num">{pct(sc.models.baseline.mean_predicted)} · {pct(sc.models.challenger.mean_predicted)}</td></tr>))}</tbody>
            </table></div>
            <h3 style={{ marginTop: 14 }}>Scenario definitions</h3>
            <div className="table-wrap"><table>
              <thead><tr><th>Scenario</th><th>Kind</th><th>Perturbation</th><th>Valid-range note</th><th className="num">Seed</th></tr></thead>
              <tbody>{s.scenarios.map((sc: Json) => (
                <tr key={sc.id}><td className="mono small nowrap">{sc.id}</td><td className="nowrap"><Badge tone="plain">{sc.kind.replace("_", " ")}</Badge></td>
                  <td className="small wide">{sc.description}</td><td className="small wide">{sc.valid_range_note}{sc.rows_below_train_min_limit ? ` ${sc.rows_below_train_min_limit} rows fall below the train minimum credit limit.` : ""}</td><td className="num">{sc.seed}</td></tr>))}</tbody>
            </table></div>
          </div>
        )}
      </Async>
      <Async state={dr}>
        {(d) => (
          <div className="card">
            <div className="sim-banner" role="note">SIMULATED batches. Derived from the tuning partition; not a live monitoring feed and not observed deployment drift.</div>
            <h2>Shift diagnostics</h2>
            <p className="small">{d.interpretation} PSI review triggers (configurable, not regulatory): watch ≥ {d.thresholds.psi_watch}, review ≥ {d.thresholds.psi_review_trigger}.</p>
            <div className="table-wrap"><table>
              <thead><tr><th>Batch</th><th>Kind</th><th>Construction</th><th className="num">Max feature PSI</th><th>Features ≥ review</th>{ROLES.map((r) => <th key={r} className="num">{ROLE_SHORT[r]} score PSI</th>)}<th className="num">Mean pred (HGB) vs observed</th>{ROLES.map((r) => <th key={r + "a"} className="num">{ROLE_SHORT[r]} AUC</th>)}</tr></thead>
              <tbody>{d.batches.map((b: Json) => (
                <tr key={b.batch}><td className="nowrap"><Badge tone="sim">{b.label}</Badge> {b.batch}</td><td className="small">{b.kind.replace(/_/g, " ")}</td><td className="small wide">{b.description}</td>
                  <td className="num">{fmt(b.max_feature_psi.psi, 3)} <span className="muted small">{b.max_feature_psi.feature}</span></td><td className="small">{b.features_review.join(", ") || "none"}</td>
                  {ROLES.map((r) => <td key={r} className="num">{fmt(b.models[r].score_psi, 3)}</td>)}
                  <td className="num">{pct(b.models.challenger.mean_predicted)} vs {pct(b.models.challenger.observed_rate)}</td>
                  {ROLES.map((r) => <td key={r + "a"} className="num">{fmt(b.models[r].roc_auc, 3)}</td>)}</tr>))}</tbody>
            </table></div>
            <p className="small muted">Labelled columns (observed rate, AUC) exist only because these batches are simulated from labelled history. A real unlabelled batch shows only the PSI and mean-prediction columns.</p>
            <DataTable summary="Sanity check: train vs tuning feature PSI (same distribution by construction)">
              <table><thead><tr><th>Feature</th><th className="num">PSI</th><th className="num">KS</th><th>Status</th></tr></thead>
                <tbody>{d.sanity_train_vs_tuning.map((f: Json) => <tr key={f.feature}><td>{f.feature}</td><td className="num">{fmt(f.psi, 4)}</td><td className="num">{fmt(f.ks_statistic, 3)}</td><td>{f.status}</td></tr>)}</tbody></table>
            </DataTable>
          </div>
        )}
      </Async>
      <MonitoringUpload studyId={studyId} />
    </div>
  );
}

function MonitoringUpload({ studyId }: { studyId: string }) {
  const [state, setState] = useState<{ phase: "idle" | "validating" | "done" | "error"; data?: Json; error?: string }>({ phase: "idle" });
  const onFile = (f?: File) => {
    if (!f) return;
    setState({ phase: "validating" });
    api.postCsv(`/api/studies/${studyId}/monitoring-batch`, f).then((d) => setState({ phase: "done", data: d })).catch((e) => setState({ phase: "error", error: e.message }));
  };
  const d = state.data;
  return (
    <div className="card">
      <h2>Compare your own batch <Badge tone="sim">user-supplied</Badge></h2>
      <p className="small">Upload a CSV with the UCI 350 columns; the target column is optional. It is validated first, compared with the train reference, and not stored.</p>
      <input type="file" accept=".csv,text/csv" aria-label="Monitoring batch CSV" onChange={(e) => onFile(e.target.files?.[0])} />
      <div aria-live="polite" style={{ marginTop: 10 }}>
        {state.phase === "validating" && <p className="small">Validating and scoring…</p>}
        {state.phase === "error" && <ErrorBox message={state.error!} />}
        {state.phase === "done" && !d.accepted && (
          <><Badge tone="crit">rejected by validation</Badge><ul className="small">{d.report.checks.filter((c: Json) => c.status === "fail").map((c: Json) => <li key={c.name}>{c.name}: {c.detail}</li>)}</ul></>
        )}
        {state.phase === "done" && d.accepted && (
          <>
            <p><Badge tone="good">validated</Badge> {d.n} rows · {d.labelled ? "labelled" : "unlabelled: performance and concept drift cannot be assessed"}</p>
            <table><thead><tr><th>Model</th><th className="num">Score PSI</th><th className="num">Mean predicted</th><th className="num">AUC</th><th className="num">Observed rate</th></tr></thead>
              <tbody>{ROLES.map((r) => { const m = d.models[r]; return <tr key={r}><td>{ROLE_SHORT[r]}</td><td className="num">{fmt(m.score_psi, 3)}</td><td className="num">{pct(m.mean_predicted)}</td><td className="num">{m.performance ? fmt(m.performance.roc_auc) : <span className="unavailable">{m.performance_unavailable ?? "no labels"}</span>}</td><td className="num">{m.performance ? pct(m.performance.prevalence) : "n/a"}</td></tr>; })}</tbody></table>
            {Object.keys(d.out_of_training_range ?? {}).length > 0 ? (
              <div className="callout warn"><strong>Values outside the training range:</strong>{" "}
                {Object.entries(d.out_of_training_range).map(([f, v]: [string, Json]) => `${f} (${v.below_train_min} below min, ${v.above_train_max} above max)`).join("; ")}. {d.range_note}</div>
            ) : <p className="small muted">No values outside the training range.</p>}
            <DataTable summary="Feature PSI"><table><tbody>{d.feature_drift.map((f: Json) => <tr key={f.feature}><td>{f.feature}</td><td className="num">{fmt(f.psi, 3)}</td><td>{f.status}</td></tr>)}</tbody></table></DataTable>
          </>
        )}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ Explanations
export function ExplanationsPage() {
  const { studyId } = useStudy();
  const ex = useApi(studyId ? `/api/studies/${studyId}/evidence/explanations` : null);
  const [role, setRole] = useState<"baseline" | "challenger">("challenger");
  const [cid, setCid] = useState<number | null>(null);
  const local = useApi(studyId && cid !== null ? `/api/studies/${studyId}/explanations/local/${cid}` : null, [cid]);
  if (!studyId) return <NoStudy />;
  return (
    <Async state={ex}>
      {(e) => {
        const m = e.models[role];
        return (
          <div className="stack">
            <div className="row">
              <Seg label="Model" value={role} onChange={setRole} options={ROLES.map((r) => ({ value: r, label: ROLE_SHORT[r] }))} />
              <span className="small muted">{e.n_explain} tuning-partition customers explained · background {e.background.n} train rows ({e.background.used_by})</span>
            </div>
            <div className="callout"><strong>What SHAP can and cannot say.</strong><ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>{e.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul></div>
            <div className="grid two">
              <div className="card">
                <h2>Global importance: mean |SHAP|</h2>
                <p className="small">{m.algorithm}. Output space: {m.output_space}. Additivity {m.additivity_ok ? <Badge tone="good">holds</Badge> : <Badge tone="crit">violated</Badge>} (max error {m.additivity_max_abs_error_calibrated.toExponential(1)}).</p>
                <BarChart title={`Mean absolute SHAP, ${ROLE_SHORT[role]}`} valueLabel="mean |SHAP| (calibrated log-odds)"
                  rows={m.global.map((g: Json) => ({ label: g.feature, values: [{ key: role, label: ROLE_SHORT[role], color: ROLE_COLOR[role], value: g.mean_abs_shap }] }))} />
                <DataTable><table><thead><tr><th>Rank</th><th>Feature</th><th className="num">mean |SHAP|</th><th className="num">Spearman(value, SHAP)</th></tr></thead>
                  <tbody>{m.global.map((g: Json) => <tr key={g.feature}><td>{g.rank}</td><td>{g.feature}</td><td className="num">{fmt(g.mean_abs_shap)}</td><td className="num">{fmt(g.value_shap_spearman, 2)}</td></tr>)}</tbody></table></DataTable>
              </div>
              <div className="card">
                <h2>Ranking stability across refits</h2>
                <p className="small">{e.stability[role].method}. Compared with the main model’s global ranking.</p>
                <table><thead><tr><th>Refit</th><th className="num">Spearman rank ρ</th><th className="num">Top-5 overlap</th><th>Top 5</th></tr></thead>
                  <tbody>{e.stability[role].refits.map((r: Json) => <tr key={r.refit}><td>{r.refit}</td><td className="num">{fmt(r.spearman_rank_correlation, 3)}</td><td className="num">{r.top5_overlap}/5</td><td className="small">{r.top5.join(", ")}</td></tr>)}</tbody></table>
                <p className="small muted" style={{ marginTop: 8 }}>Refits resample the training data only. They show sensitivity of the explanation to training noise, not its correctness.</p>
              </div>
            </div>
            <div className="card">
              <h2>Local explanation</h2>
              <label className="field" style={{ maxWidth: 320 }}><span>Customer ID (development explanation sample)</span>
                <select value={cid ?? ""} onChange={(ev) => setCid(ev.target.value ? Number(ev.target.value) : null)}>
                  <option value="">Choose a customer…</option>
                  {e.explained_ids.map((id: number) => <option key={id} value={id}>{id}</option>)}
                </select>
              </label>
              {cid !== null && (
                <Async state={local}>
                  {(l) => { const lm = l.models[role]; const top = lm.contributions.slice(0, 10); return (
                    <div style={{ marginTop: 12 }}>
                      <p>Base value {fmt(lm.base_value_calibrated_logit, 3)} + Σ SHAP {signed(lm.calibrated_logit - lm.base_value_calibrated_logit, 3)} = calibrated log-odds {fmt(lm.calibrated_logit, 3)} → probability <strong>{pct(lm.calibrated_probability, 1)}</strong></p>
                      <BarChart title="Top local contributions" valueLabel="SHAP contribution (calibrated log-odds; right = raises predicted risk)"
                        rows={top.map((c: Json) => ({ label: `${c.feature} = ${Number(c.value).toLocaleString()}`, values: [{ key: role, label: ROLE_SHORT[role], color: ROLE_COLOR[role], value: c.shap }] }))} />
                      <p className="small muted">This attributes the model’s score, not the customer’s real-world reasons. It is not an adverse-action notice.</p>
                    </div>); }}
                </Async>
              )}
            </div>
          </div>
        );
      }}
    </Async>
  );
}

// ------------------------------------------------------------------ Review
export function ReviewPage() {
  const { studyId, refresh } = useStudy();
  const rv = useApi(studyId ? `/api/studies/${studyId}/review` : null);
  const [actor, setActor] = useActor();
  const [prefill, setPrefill] = useState<Json>(null);
  if (!studyId) return <NoStudy />;
  const reload = () => { rv.reload(); refresh(); };
  return (
    <Async state={rv}>
      {(r) => (
        <div className="stack">
          <div className="card">
            <div className="row">
              <h2 style={{ margin: 0 }}>Review case</h2><StateBadge state={r.case.state} />
              <span className="small muted">version {r.case.version} · evidence <span className="mono">{(r.case.evidence_sha256 ?? "").slice(0, 16)}…</span></span>
              {r.case.evidence_current === false && <Badge tone="crit">evidence changed since last case update</Badge>}
              {r.stale_reasons.length > 0 && <Badge tone="crit">stale: {r.stale_reasons.join("; ")}</Badge>}
            </div>
            <p className="small" style={{ marginTop: 8 }}>Candidate: <strong>{r.case.candidate_role ? ROLE_LABEL[r.case.candidate_role] : "—"}</strong>. Acceptance means “accepted for this local research prototype”, never production, regulatory or bank approval. {r.identity_note}</p>
            <label className="field" style={{ maxWidth: 320 }}><span>Your name (recorded with every action; not authenticated)</span><input type="text" value={actor} onChange={(e) => setActor(e.target.value)} /></label>
          </div>
          <div className="card">
            <h2>Deterministic checks</h2>
            <p className="small muted">Configured checks flag concerns for a human. None of them can accept or reject a model.</p>
            <div className="table-wrap"><table>
              <thead><tr><th>Status</th><th>Check</th><th>Result</th><th>Evidence</th><th /></tr></thead>
              <tbody>{r.checks.map((c: Json) => (
                <tr key={c.check_id}><td><CheckBadge status={c.status} /></td><td className="mono small nowrap">{c.check_id}</td><td className="small">{c.message}</td><td className="mono small">{c.evidence_ref}</td>
                  <td>{c.status === "flag" && !r.findings.some((f: Json) => f.source_check === c.check_id) && !["accepted_for_research", "rejected"].includes(r.case.state) && (
                    <button className="btn" onClick={() => setPrefill({ ...c, key: Date.now() })}>Raise finding</button>)}</td></tr>))}</tbody>
            </table></div>
          </div>
          <FindingForm key={prefill?.key ?? "blank"} studyId={studyId} actor={actor} prefill={prefill} r={r} onDone={reload} />
          <div className="card">
            <h2>Findings ({r.findings.length})</h2>
            {r.findings.length === 0 && <p className="muted small">None recorded.</p>}
            {r.findings.map((f: Json) => <FindingItem key={f.finding_id} f={f} actor={actor} statuses={r.finding_statuses} onDone={reload} />)}
            <DataTable summary="Severity rubric"><table><tbody>{Object.entries(r.severity_rubric).map(([k, v]) => <tr key={k}><th style={{ width: 90 }}>{k}</th><td className="small">{String(v)}</td></tr>)}</tbody></table></DataTable>
          </div>
          <DecisionForm studyId={studyId} r={r} actor={actor} onDone={reload} />
          <div className="card">
            <h2>Event history</h2>
            <p className="small">Append-only, hash-chained: chain {r.event_chain.ok ? <Badge tone="good">intact</Badge> : <Badge tone="crit">broken at event {r.event_chain.broken_at_event}</Badge>} ({r.event_chain.n_events} events). A local SQLite file is not tamper-proof.</p>
            <ol className="timeline">{[...r.events].reverse().map((e: Json) => {
              const p = JSON.parse(e.payload_json);
              return <li key={e.event_id}><strong>{e.event_type.replace(/_/g, " ")}</strong> <span className="muted">{e.from_state ?? "—"} → {e.to_state ?? "—"}</span><div className="small muted">{e.at_utc} · {e.actor} ({e.actor_role}{e.identity_verified ? "" : ", unverified"})</div>{p.rationale && <div className="small">“{p.rationale}”</div>}{p.title && <div className="small">{p.severity}: {p.title}</div>}{p.reason && <div className="small">{p.reason}</div>}</li>;
            })}</ol>
          </div>
        </div>
      )}
    </Async>
  );
}

function FindingForm({ studyId, actor, prefill, r, onDone }: { studyId: string; actor: string; prefill: Json; r: Json; onDone: () => void }) {
  const [f, setF] = useState({ category: prefill?.category ?? "calibration", severity: prefill?.suggested_severity ?? "medium", title: prefill ? `Check ${prefill.check_id} flagged` : "", description: prefill?.message ?? "", evidence_ref: prefill?.evidence_ref ?? "", owner: "" });
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({});
  const [busy, setBusy] = useState(false);
  const closed = ["accepted_for_research", "rejected"].includes(r.case.state);
  const submit = () => {
    setBusy(true);
    setMsg({});
    api.post(`/api/studies/${studyId}/findings`, { ...f, actor, source_check: prefill?.check_id ?? null })
      .then((x) => { setMsg({ ok: `Finding #${x.finding_id} recorded.` }); onDone(); })
      .catch((e) => setMsg({ err: e.message })).finally(() => setBusy(false));
  };
  return (
    <div className="card">
      <h2>Record a finding</h2>
      {closed ? <p className="small muted">The case is closed; record new findings in a new study.</p> : (
        <div className="form-grid">
          <label className="field"><span>Category</span><select value={f.category} onChange={(e) => setF({ ...f, category: e.target.value })}>{r.categories.map((c: string) => <option key={c}>{c}</option>)}</select></label>
          <label className="field"><span>Severity</span><select value={f.severity} onChange={(e) => setF({ ...f, severity: e.target.value })}>{Object.keys(r.severity_rubric).map((s) => <option key={s}>{s}</option>)}</select></label>
          <label className="field full"><span>Title</span><input type="text" value={f.title} onChange={(e) => setF({ ...f, title: e.target.value })} /></label>
          <label className="field full"><span>Description</span><textarea value={f.description} onChange={(e) => setF({ ...f, description: e.target.value })} /></label>
          <label className="field"><span>Evidence reference</span><input type="text" value={f.evidence_ref} onChange={(e) => setF({ ...f, evidence_ref: e.target.value })} placeholder="e.g. development.json#/evaluation/challenger" /></label>
          <label className="field"><span>Owner</span><input type="text" value={f.owner} onChange={(e) => setF({ ...f, owner: e.target.value })} /></label>
          <div className="full row"><button className="btn primary" disabled={busy || !actor.trim()} onClick={submit}>{busy ? "Saving…" : "Record finding"}</button>{!actor.trim() && <span className="small muted">Enter your name above first.</span>}
            {msg.ok && <span className="small" role="status">{msg.ok}</span>}{msg.err && <span className="small" role="alert" style={{ color: "var(--critical-text)" }}>{msg.err}</span>}</div>
        </div>
      )}
    </div>
  );
}

function FindingItem({ f, actor, statuses, onDone }: { f: Json; actor: string; statuses: string[]; onDone: () => void }) {
  const [status, setStatus] = useState(f.status);
  const [comment, setComment] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const save = () => {
    setErr(null);
    api.patch(`/api/findings/${f.finding_id}`, { actor, status: status === f.status ? null : status, comment: comment || null }).then(() => { setComment(""); onDone(); }).catch((e) => setErr(e.message));
  };
  const tone = f.severity === "critical" || f.severity === "high" ? "crit" : f.severity === "medium" ? "warn" : "neutral";
  return (
    <div style={{ borderTop: "1px solid var(--grid)", padding: "10px 0" }}>
      <div className="row"><strong>#{f.finding_id} {f.title}</strong><Badge tone={tone}>{f.severity}</Badge><Badge tone="plain">{f.category}</Badge><Badge tone={["open", "in_remediation"].includes(f.status) ? "warn" : "good"}>{f.status.replace(/_/g, " ")}</Badge></div>
      <p className="small" style={{ margin: "4px 0" }}>{f.description}</p>
      <p className="small muted">Evidence <span className="mono">{f.evidence_ref}</span> · owner {f.owner} · raised by {f.created_by} {f.created_utc}{f.reviewer_comment ? ` · latest comment: “${f.reviewer_comment}”` : ""}</p>
      <details><summary>History and update ({f.history.length} events)</summary>
        <ul className="small">{f.history.map((h: Json) => { const p = JSON.parse(h.payload_json); return <li key={h.id}>{h.at_utc} {h.actor}: {h.event_type} {p.from_status ? `${p.from_status} → ${p.to_status}` : ""} {p.comment ?? ""} {p.remediation ?? ""}</li>; })}</ul>
        <div className="row">
          <select aria-label="Finding status" value={status} onChange={(e) => setStatus(e.target.value)} style={{ width: 180 }}>{statuses.map((s) => <option key={s}>{s}</option>)}</select>
          <input type="text" aria-label="Comment or remediation" placeholder="Comment / remediation note (required to close)" value={comment} onChange={(e) => setComment(e.target.value)} style={{ flex: 1, minWidth: 200 }} />
          <button className="btn" disabled={!actor.trim()} onClick={save}>Update</button>
        </div>
        {err && <p className="small" role="alert" style={{ color: "var(--critical-text)" }}>{err}</p>}
      </details>
    </div>
  );
}

const ACTION_LABEL: Record<string, string> = { start_review: "Start review", accept: "Accept for research", request_changes: "Request changes", reject: "Reject", reopen: "Reopen review" };

function DecisionForm({ studyId, r, actor, onDone }: { studyId: string; r: Json; actor: string; onDone: () => void }) {
  const [action, setAction] = useState<string>(r.allowed_actions[0] ?? "");
  const [rationale, setRationale] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [requestId, setRequestId] = useState(newRequestId);
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({});
  const [busy, setBusy] = useState(false);
  useEffect(() => { setAction(r.allowed_actions[0] ?? ""); }, [r.allowed_actions.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!r.allowed_actions.length) return <div className="card"><h2>Decision</h2><p className="small muted">No further transitions from “{r.case.state.replace(/_/g, " ")}”.</p></div>;
  const submit = () => {
    setBusy(true);
    setMsg({});
    api.post(`/api/studies/${studyId}/review/transition`, { action, actor, rationale, confirm, expected_version: r.case.version, expected_evidence_sha256: r.case.evidence_sha256, request_id: requestId })
      .then((c) => { setMsg({ ok: `Recorded: case is now ${c.state.replace(/_/g, " ")}.` }); setRationale(""); setConfirm(false); setRequestId(newRequestId()); onDone(); })
      .catch((e) => setMsg({ err: e.message })).finally(() => setBusy(false));
  };
  return (
    <div className="card">
      <h2>Human decision</h2>
      <p className="small">Submitted with the case version ({r.case.version}) and evidence hash you are looking at. If either changed, the server rejects it as stale; a resubmission of the same request is rejected as a duplicate.</p>
      <div className="form-grid">
        <label className="field"><span>Action</span><select value={action} onChange={(e) => setAction(e.target.value)}>{r.allowed_actions.map((a: string) => <option key={a} value={a}>{ACTION_LABEL[a] ?? a}</option>)}</select></label>
        <div />
        <label className="field full"><span>Rationale (at least 20 characters; cite evidence and limitations)</span><textarea value={rationale} onChange={(e) => setRationale(e.target.value)} /></label>
        <label className="check full"><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} />
          <span>I confirm this is my decision for a local research prototype, made after reviewing the evidence above. It is not a production, regulatory or bank approval.</span></label>
        {action === "accept" && !r.final_evaluated && <div className="full callout warn">Acceptance requires the one-shot final-holdout audit to have been run.</div>}
        <div className="full row"><button className={`btn ${action === "reject" ? "danger" : "primary"}`} disabled={busy || !confirm || !actor.trim() || rationale.trim().length < 20} onClick={submit}>{busy ? "Submitting…" : ACTION_LABEL[action] ?? action}</button>
          {msg.ok && <span className="small" role="status">{msg.ok}</span>}{msg.err && <span className="small" role="alert" style={{ color: "var(--critical-text)" }}>{msg.err}</span>}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ Runs & reports
export function RunsPage() {
  const { status, studyId, study, refresh } = useStudy();
  const jobs = useApi("/api/jobs", [status?.active_job?.status, status?.studies?.length]);
  const policy = useApi("/api/policy");
  const detail = useApi(studyId ? `/api/studies/${studyId}` : null, [study?.status]);
  const [actor, setActor] = useActor();
  const [confirmDev, setConfirmDev] = useState(false);
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({});
  const active = status?.active_job;
  const startDev = () => {
    setMsg({});
    api.post("/api/jobs/development-study", { actor, confirm: confirmDev }).then((r) => { setMsg({ ok: r.cached ? r.message : `Started ${r.job_id} for ${r.study_id}.` }); setConfirmDev(false); refresh(); jobs.reload(); }).catch((e) => setMsg({ err: e.message }));
  };
  const lastFailed = useMemo(() => (jobs.data ?? []).find((j: Json) => j.status === "failed" || j.status === "interrupted"), [jobs.data]);
  return (
    <div className="stack">
      <div className="grid two">
        <div className="card">
          <h2>Start a development study</h2>
          <p className="small">Trains both models on the train partition, calibrates, evaluates on tuning and freezes the champion. Never touches the final holdout. If an identical dataset/policy/code study exists, it is reused instead of retrained.</p>
          <div className="form-grid">
            <label className="field full"><span>Your name</span><input type="text" value={actor} onChange={(e) => setActor(e.target.value)} /></label>
            <label className="check full"><input type="checkbox" checked={confirmDev} onChange={(e) => setConfirmDev(e.target.checked)} /><span>Start training now (bounded CPU job; a few minutes on a laptop)</span></label>
            <div className="full row"><button className="btn primary" disabled={!!active || !confirmDev || !actor.trim()} onClick={startDev}>Start development study</button>{active && <span className="small muted">A job is already {active.status}.</span>}</div>
          </div>
          {msg.ok && <p className="small" role="status">{msg.ok}</p>}{msg.err && <div style={{ marginTop: 8 }}><ErrorBox message={msg.err} /></div>}
          {active && (
            <div style={{ marginTop: 12 }} aria-live="polite"><p className="small"><strong>{active.kind}</strong> {active.status}: {active.progress ?? "queued"}</p><div className="progress"><span /></div></div>
          )}
          {!active && lastFailed && <div className="callout crit" style={{ marginTop: 12 }}><strong>Last failed job {lastFailed.job_id}:</strong><pre className="small" style={{ whiteSpace: "pre-wrap", margin: "6px 0 0" }}>{(lastFailed.error ?? "").split("\n")[0]}</pre></div>}
        </div>
        {study ? <FinalPanel study={study} actor={actor} active={!!active} onDone={() => { refresh(); jobs.reload(); }} /> : <Empty title="No study selected" />}
      </div>
      {study && (
        <div className="card">
          <h2>Downloads for {study.study_id}</h2>
          <div className="row">
            <a className="btn" href={`/api/studies/${study.study_id}/report.md`} download>Validation report (Markdown)</a>
            <a className="btn" href={`/api/studies/${study.study_id}/results.json`} download>Machine-readable results (JSON)</a>
          </div>
          <Async state={detail}>{(d) => (
            <>
              <h3 style={{ marginTop: 14 }}>Evidence files (SHA-256 verified on every read)</h3>
              <div className="table-wrap"><table><thead><tr><th>Evidence</th><th>SHA-256</th><th>Written</th></tr></thead>
                <tbody>{d.evidence.map((e: Json) => <tr key={e.name}><td>{e.name}</td><td className="mono small">{e.sha256.slice(0, 24)}…</td><td className="small">{e.created_utc}</td></tr>)}</tbody></table></div>
              <h3 style={{ marginTop: 14 }}>Model artifacts</h3>
              <table><thead><tr><th>Role</th><th>Family</th><th>Artifact SHA-256</th><th>Preprocessing</th></tr></thead>
                <tbody>{d.models.map((m: Json) => <tr key={m.role}><td>{m.role}</td><td>{m.family}</td><td className="mono small">{m.artifact_sha256.slice(0, 24)}…</td><td className="small">{m.preprocessing_version}</td></tr>)}</tbody></table>
              <h3 style={{ marginTop: 14 }}>Final-holdout access log</h3>
              {d.holdout_access_log.length === 0 ? <p className="small muted">No access attempts.</p> : <table><thead><tr><th>When</th><th>Actor</th><th>Outcome</th></tr></thead><tbody>{d.holdout_access_log.map((h: Json) => <tr key={h.id}><td className="small">{h.at_utc}</td><td>{h.actor}</td><td>{h.outcome}</td></tr>)}</tbody></table>}
            </>
          )}</Async>
        </div>
      )}
      <div className="card">
        <h2>All studies</h2>
        <div className="table-wrap"><table>
          <thead><tr><th>Study</th><th>Status</th><th>Champion</th><th>Review</th><th>Identity</th><th>Created</th></tr></thead>
          <tbody>{(status?.studies ?? []).map((s: Json) => (
            <tr key={s.study_id}><td className="mono small">{s.study_id}</td><td>{s.status}{s.error && <div className="small" style={{ color: "var(--critical-text)" }}>{s.error}</div>}</td><td>{s.champion ? ROLE_SHORT[s.champion] : "—"}</td><td><StateBadge state={s.review_state} /></td>
              <td>{s.stale_reasons.length ? <Badge tone="crit">stale: {s.stale_reasons.join("; ")}</Badge> : <Badge tone="good">current</Badge>}</td><td className="small">{s.created_utc}</td></tr>))}</tbody>
        </table></div>
      </div>
      <div className="card">
        <h2>Jobs</h2>
        <Async state={jobs}>{(js) => js.length === 0 ? <p className="small muted">No jobs yet.</p> : (
          <div className="table-wrap"><table><thead><tr><th>Job</th><th>Kind</th><th>Status</th><th>Progress / error</th><th>By</th><th>Created</th></tr></thead>
            <tbody>{js.map((j: Json) => <tr key={j.job_id}><td className="mono small">{j.job_id}</td><td>{j.kind}</td><td><CheckBadge status={j.status === "succeeded" ? "pass" : j.status === "failed" || j.status === "interrupted" ? "fail" : "pending"} /> {j.status}</td><td className="small">{j.error ? j.error.split("\n")[0] : j.progress}</td><td>{j.requested_by}</td><td className="small">{j.created_utc}</td></tr>)}</tbody></table></div>
        )}</Async>
      </div>
      <Async state={policy}>{(p) => (
        <div className="card">
          <h2>Pre-registered policy summary</h2>
          <p className="small mono">version {p.policy.policy_version} · {p.policy.status} · SHA-256 {p.sha256}</p>
          <table><tbody>
            {[["Split", `${Object.entries(p.policy.split.fractions).map(([k, v]) => `${k} ${(Number(v) * 100).toFixed(0)}%`).join(", ")}; seed ${p.policy.split.seed}`], ["Demographic predictors", String(p.policy.features.demographic_predictors)], ["Hyperparameters", p.policy.models.hyperparameter_selection], ["Calibration", p.policy.calibration.selection_rule], ["Champion rule", p.policy.champion_selection.rule], ["Threshold", `${p.policy.threshold.primary}: ${p.policy.threshold.rationale}`], ["Uncertainty", `${p.policy.uncertainty.method}; ${p.policy.uncertainty.n_resamples} resamples`], ["Final holdout", p.policy.final_holdout_rule], ["Approval", p.policy.release_approval]].map(([k, v]) => <tr key={k}><th style={{ width: 170 }}>{k}</th><td className="small">{v}</td></tr>)}
          </tbody></table>
          <DataTable summary="Raw policy JSON"><pre className="small mono" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(p.policy, null, 2)}</pre></DataTable>
        </div>
      )}</Async>
    </div>
  );
}

function FinalPanel({ study, actor, active, onDone }: { study: Json; actor: string; active: boolean; onDone: () => void }) {
  const [ack, setAck] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({});
  const run = () => {
    setMsg({});
    api.post(`/api/studies/${study.study_id}/final-evaluation`, { actor, confirm_final_holdout_access: confirm, acknowledgement: ack })
      .then((r) => { setMsg({ ok: `Started ${r.job_id}. Watch the Jobs table for the result.` }); setConfirm(false); onDone(); }).catch((e) => setMsg({ err: e.message }));
  };
  return (
    <div className="card">
      <h2>Final-holdout audit</h2>
      {study.final_evaluated ? (
        <p>Already evaluated once for this study. <Badge tone="good">audit complete</Badge> Results are on the comparison and segment pages. A redesign needs a new study with fresh unseen data, or must be labelled exploratory.</p>
      ) : study.stale_reasons.length ? (
        <div className="callout crit">This study is stale ({study.stale_reasons.join("; ")}). The final holdout cannot be opened for it; start a new development study.</div>
      ) : (
        <>
          <div className="callout warn"><strong>One shot.</strong> The selection record is frozen. The holdout is evaluated once and the access is logged. Whatever it shows becomes an audit finding; it must not be used to change models, calibration, thresholds or criteria.</div>
          <div className="form-grid">
            <label className="field full"><span>Type an acknowledgement containing the word “frozen”</span><input type="text" value={ack} onChange={(e) => setAck(e.target.value)} placeholder="I accept the selection is frozen" /></label>
            <label className="check full"><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} /><span>I confirm final-holdout access for {study.study_id} (champion: {ROLE_SHORT[study.champion] ?? "—"}).</span></label>
            <div className="full"><button className="btn primary" disabled={active || !confirm || !actor.trim() || !ack.toLowerCase().includes("frozen")} onClick={run}>Run final audit</button></div>
          </div>
          <p className="small muted">This is a local, student-operated lock: anyone with file access could read the data directly. The log records access through this application only.</p>
        </>
      )}
      {msg.ok && <p className="small" role="status">{msg.ok}</p>}{msg.err && <div style={{ marginTop: 8 }}><ErrorBox message={msg.err} /></div>}
    </div>
  );
}
