import { ReactNode, useCallback, useEffect, useState } from "react";
import { api, Json } from "./api";

export function useApi(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<Json>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const load = useCallback(() => {
    if (!path) {
      setData(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    api
      .get(path)
      .then((d) => setData(d))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => load(), [load]);
  return { data, error, loading, reload: load };
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite">
      <span className="muted small">{label}…</span>
      <div className="skeleton" style={{ marginTop: 8 }} />
    </div>
  );
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error" role="alert">
      <strong>Could not load:</strong> {message}{" "}
      {onRetry && (
        <button className="btn" style={{ marginLeft: 8 }} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="card empty">
      <h3>{title}</h3>
      <div className="small">{children}</div>
    </div>
  );
}

/** Wraps an API call with loading / error / empty states. */
export function Async({ state, children, empty }: { state: { data: Json; error: string | null; loading: boolean; reload: () => void }; children: (d: Json) => ReactNode; empty?: ReactNode }) {
  if (state.loading && !state.data) return <Loading />;
  if (state.error) return <ErrorBox message={state.error} onRetry={state.reload} />;
  if (!state.data) return <>{empty ?? null}</>;
  return <>{children(state.data)}</>;
}

export function Badge({ tone = "plain", children, title }: { tone?: "good" | "warn" | "serious" | "crit" | "plain" | "sim" | "neutral"; children: ReactNode; title?: string }) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {children}
    </span>
  );
}

const CHECK_TONE: Record<string, "good" | "warn" | "crit" | "neutral"> = { pass: "good", ok: "good", info: "neutral", warn: "warn", flag: "warn", fail: "crit", pending: "neutral" };
export function CheckBadge({ status }: { status: string }) {
  const tone = CHECK_TONE[status] ?? "neutral";
  return <Badge tone={tone}>{status}</Badge>;
}

const STATE_TONE: Record<string, "good" | "warn" | "crit" | "neutral"> = {
  draft: "neutral", evaluated: "neutral", in_review: "warn", accepted_for_research: "good", changes_requested: "warn", rejected: "crit",
};
export function StateBadge({ state }: { state: string | null | undefined }) {
  if (!state) return null;
  return <Badge tone={STATE_TONE[state] ?? "neutral"}>{state.replace(/_/g, " ")}</Badge>;
}

export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function NoStudy() {
  return (
    <Empty title="No study yet">
      Nothing has been trained. Opening the dashboard never trains a model. Start a development study from{" "}
      <a href="#/runs">Reports &amp; runs</a> when you are ready.
    </Empty>
  );
}
