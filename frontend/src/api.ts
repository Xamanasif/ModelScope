// Thin fetch wrapper. Evidence payloads are typed loosely (`any`) on purpose:
// they are produced by the Python evidence writers, which are the schema of record.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Json = any;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle(res: Response): Promise<Json> {
  const text = await res.text();
  let body: Json = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    let detail: string = (body && body.detail) || res.statusText || "Request failed";
    if (Array.isArray(detail)) detail = detail.map((d: Json) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
    throw new ApiError(res.status, String(detail));
  }
  return body;
}

export const api = {
  get: (path: string) => fetch(path).then(handle),
  post: (path: string, body: unknown) =>
    fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
  patch: (path: string, body: unknown) =>
    fetch(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
  postCsv: (path: string, file: File) =>
    fetch(path, { method: "POST", headers: { "Content-Type": "text/csv" }, body: file }).then(handle),
};

export const ROLE_LABEL: Record<string, string> = {
  baseline: "Baseline · logistic regression",
  challenger: "Challenger · gradient boosting",
};
export const ROLE_SHORT: Record<string, string> = { baseline: "Baseline (LR)", challenger: "Challenger (HGB)" };
export const ROLE_COLOR: Record<string, string> = { baseline: "var(--series-1)", challenger: "var(--series-2)" };

export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "n/a";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}
export function pct(v: unknown, digits = 1): string {
  return typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(digits)}%` : "n/a";
}
export function signed(v: unknown, digits = 4): string {
  return typeof v === "number" && Number.isFinite(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}` : "n/a";
}
export function ci(c: unknown, digits = 4): string {
  return Array.isArray(c) && c[0] !== null && c[0] !== undefined ? `[${fmt(c[0], digits)}, ${fmt(c[1], digits)}]` : "n/a";
}

export function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
export function storageSet(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* per-viewer convenience only */
  }
}
