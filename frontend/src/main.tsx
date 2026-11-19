import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { api, Json, storageGet, storageSet } from "./api";
import { StudyContext } from "./ctx";
import { Badge, ErrorBox, Loading, StateBadge } from "./ui";
import { ComparisonPage, DataPage, OverviewPage, SegmentsPage } from "./pages1";
import { ExplanationsPage, ReviewPage, RunsPage, StressPage } from "./pages2";


const PAGES = [
  { path: "overview", label: "Overview", el: OverviewPage },
  { path: "data", label: "Data quality", el: DataPage },
  { path: "comparison", label: "Model comparison", el: ComparisonPage },
  { path: "segments", label: "Segment analysis", el: SegmentsPage },
  { path: "stress", label: "Stress & shift lab", el: StressPage },
  { path: "explanations", label: "Explanations", el: ExplanationsPage },
  { path: "review", label: "Review workspace", el: ReviewPage },
  { path: "runs", label: "Reports & runs", el: RunsPage },
];

function useHashRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("?")[0] || "overview";
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

function App() {
  const route = useHashRoute();
  const [status, setStatus] = useState<Json>(null);
  const [error, setError] = useState<string | null>(null);
  const [studyId, setStudyIdState] = useState<string | null>(storageGet("modelscope.study"));
  const [menuOpen, setMenuOpen] = useState(false);
  const refresh = useCallback(() => {
    api.get("/api/status").then((s) => { setStatus(s); setError(null); }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => refresh(), [refresh]);
  useEffect(() => {
    if (!status?.active_job) return;
    const t = window.setInterval(refresh, 2000);
    return () => window.clearInterval(t);
  }, [status?.active_job, refresh]);
  const studies: Json[] = status?.studies ?? [];
  const usable = studies.filter((s) => s.status !== "failed" && s.status !== "running");
  const selected = usable.find((s) => s.study_id === studyId) ?? usable[0] ?? null;
  const setStudyId = (id: string) => { storageSet("modelscope.study", id); setStudyIdState(id); };
  const page = PAGES.find((p) => p.path === route) ?? PAGES[0];
  const Page = page.el;
  useEffect(() => { document.title = `${page.label} · ModelScope`; setMenuOpen(false); }, [page.label]);

  return (
    <StudyContext.Provider value={{ status, studyId: selected?.study_id ?? null, study: selected, setStudyId, refresh }}>
      <div className="layout">
        <aside className={`sidebar${menuOpen ? " open" : ""}`}>
          <button className="btn menu-btn" aria-expanded={menuOpen} aria-controls="nav" onClick={() => setMenuOpen(!menuOpen)}>Menu</button>
          <div className="brand">ModelScope<small>Credit-risk model validation · local research prototype</small></div>
          <nav aria-label="Sections">
            <ul className="nav" id="nav">
              {PAGES.map((p) => (
                <li key={p.path}><a href={`#/${p.path}`} aria-current={p.path === page.path ? "page" : undefined}>{p.label}</a></li>
              ))}
            </ul>
          </nav>
          <div className="foot">Not production software. No bank, regulatory or independent approval is implied.</div>
        </aside>
        <main className="main">
          <div className="topbar">
            <div>
              <h1>{page.label}</h1>
              <div className="sub">UCI “Default of Credit Card Clients” (ID 350) · baseline logistic regression vs gradient-boosting challenger</div>
            </div>
            <div className="row">
              {status?.active_job && <Badge tone="warn">job {status.active_job.status}: {status.active_job.progress ?? status.active_job.kind}</Badge>}
              {usable.length > 0 && (
                <label className="field" style={{ minWidth: 260 }}>
                  <span className="small">Study</span>
                  <select value={selected?.study_id ?? ""} onChange={(e) => setStudyId(e.target.value)}>
                    {usable.map((s) => (
                      <option key={s.study_id} value={s.study_id}>{s.study_id}{s.stale_reasons.length ? " (stale)" : ""}</option>
                    ))}
                  </select>
                </label>
              )}
              {selected && <StateBadge state={selected.review_state} />}
              {selected?.stale_reasons?.length > 0 && <Badge tone="crit" title={selected.stale_reasons.join("; ")}>stale evidence</Badge>}
            </div>
          </div>
          {error && <ErrorBox message={`API unavailable (${error}). Is uvicorn running on 127.0.0.1:8000?`} onRetry={refresh} />}
          {!status && !error ? <Loading label="Connecting to API" /> : status && <Page />}
        </main>
      </div>
    </StudyContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
