# API reference

FastAPI app `app/main.py`, bound to `127.0.0.1` by `run.ps1`. Interactive
OpenAPI docs: http://127.0.0.1:8000/docs (schema at `/openapi.json`). Errors
return `{"detail": "..."}` with 400 (bad request / missing confirmation), 404,
409 (conflict: running job, stale or duplicate decision, sealed or already-used
holdout), 413 (upload too large), 422 (validation) or 500 (integrity check
failed: tampered evidence or model artifact).

## Status and data
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/status` | dataset presence, source manifest, current identity (dataset/policy/code hashes), studies with stale reasons, active job |
| GET | `/api/policy` | pre-registered policy and its SHA-256 |
| GET | `/api/data/quality` | validation report, split counts/prevalence/overlaps, manifest hash, code semantics |
| POST | `/api/data/validate-upload` | body = raw CSV (`Content-Type: text/csv`, ≤ 20 MB). Validates only; never stored or trained |

## Jobs (never started by page loads)
| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/api/jobs/development-study` | `{"actor": str, "confirm": true}` | 202 `{job_id, study_id}`; or `{cached: true, study_id}` if an identical identity exists; 409 if a job is active |
| POST | `/api/studies/{id}/final-evaluation` | `{"actor", "confirm_final_holdout_access": true, "acknowledgement": "...frozen..."}` | one-shot audit job; refusals are logged and the job fails with the reason |
| GET | `/api/jobs`, `/api/jobs/{job_id}` | | status `queued/running/succeeded/failed/interrupted`, progress text, error text |

## Studies and evidence
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/studies` | list with stale reasons, champion, review state |
| GET | `/api/studies/{id}` | evidence index (name, SHA-256), models, holdout access log |
| GET | `/api/studies/{id}/evidence/{name}` | `data_quality, split_manifest, development, selection, slices_tuning, stress, drift, explanations, final, slices_final` (hash-verified on read) |
| GET | `/api/studies/{id}/explanations/local/{customer_id}` | local SHAP for a customer in the development explanation sample |
| GET | `/api/studies/{id}/threshold-scenario?cost_fp=&cost_fn=&partition_name=tuning\|final` | user-supplied cost scenario; final only after the audit |
| POST | `/api/studies/{id}/monitoring-batch` | body = CSV (labels optional); validation then drift vs train reference; not persisted |
| GET | `/api/studies/{id}/report.md` | Markdown validation report (download) |
| GET | `/api/studies/{id}/results.json` | machine-readable evidence bundle (download) |

## Review
| Method | Path | Body |
|---|---|---|
| GET | `/api/studies/{id}/review` | case, checks, findings (+history), events, chain verification, allowed actions, rubric |
| POST | `/api/studies/{id}/review/transition` | `{action: start_review\|accept\|request_changes\|reject\|reopen, actor, rationale (≥20 chars), confirm: true, expected_version, expected_evidence_sha256, request_id}` |
| POST | `/api/studies/{id}/findings` | `{actor, category, severity, title, description, evidence_ref, owner, source_check?}` |
| PATCH | `/api/findings/{finding_id}` | `{actor, status?, comment?, remediation?}`; closing statuses need a comment or remediation |

## PowerShell examples
```powershell
$base = "http://127.0.0.1:8000"
Invoke-RestMethod "$base/api/status" | Select-Object dataset_present, policy_version
$job = Invoke-RestMethod -Method Post "$base/api/jobs/development-study" -ContentType "application/json" `
       -Body (@{actor="Your Name"; confirm=$true} | ConvertTo-Json)
Invoke-RestMethod "$base/api/jobs/$($job.job_id)"
Invoke-RestMethod -Method Post "$base/api/data/validate-upload" -ContentType "text/csv" -InFile .\some.csv
Invoke-WebRequest "$base/api/studies/$($job.study_id)/report.md" -OutFile report.md
```
