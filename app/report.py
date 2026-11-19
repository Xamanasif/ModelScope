"""Markdown validation report rendered ONLY from registered evidence files."""
from __future__ import annotations

import json


def _f(v, digits=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(ci, digits=4):
    return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def build_report(study: dict, stale: list, evidence: dict, case: dict, findings: list, events: list,
                 checks: list, chain: dict) -> str:
    dev, fin = evidence.get("development"), evidence.get("final")
    sel, q = evidence.get("selection"), evidence.get("data_quality")
    L = []
    w = L.append
    w(f"# ModelScope validation report — {study['study_id']}")
    w("")
    w("> Local research prototype. Not a production deployment, bank approval, regulatory compliance "
      "statement or independent organisational validation.")
    w("")
    w(f"- Status: **{study['status']}**; review state: **{case['state'] if case else 'n/a'}**")
    w(f"- Identity: dataset `{study['dataset_sha256'][:16]}…`, policy `{study['policy_sha256'][:16]}…`, "
      f"code `{study['code_source_sha256'][:16]}…`, git `{study['git_revision'] or 'not a git checkout'}`")
    w(f"- Stale: {'YES — ' + '; '.join(stale) if stale else 'no'}")
    w(f"- Split manifest: `{study['split_manifest_sha256']}`")
    w("")
    if q:
        w("## Data quality")
        w("| check | status | detail |")
        w("|---|---|---|")
        for c in q["checks"]:
            w(f"| {c['name']} | {c['status']} | {c['detail'].replace('|', '/')} |")
        w("")
    if dev:
        w("## Development evidence (tuning partition)")
        w(f"Partition sizes: {dev['partition_sizes']}; prevalence: "
          + ", ".join(f"{k} {v:.4f}" for k, v in dev["prevalence"].items()))
        w("")
        w("| model | selected params | calibration | AUC | AP | log loss | Brier | ECE | CITL | slope |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for role, m in dev["models"].items():
            ev = dev["evaluation"][role]["calibrated"]
            met, cal = ev["metrics"], ev["calibration"]
            w(f"| {role} ({m['family']}) | `{json.dumps(m['selected_params'])}` | {m['calibration']['method']} "
              f"(a={m['calibration']['a']:.3f}, b={m['calibration']['b']:.3f}) | {_f(met['roc_auc'])} | "
              f"{_f(met['average_precision'])} | {_f(met['log_loss'])} | {_f(met['brier'])} | "
              f"{_f(cal['ece_quantile_10'])} | {_f(cal['calibration_in_the_large'])} | {_f(cal['slope'], 3)} |")
        w("")
        w("Untuned vs selected (uncalibrated, tuning partition):")
        w("")
        w("| model | untuned AUC | untuned log loss | selected AUC | selected log loss | search s | configs |")
        w("|---|---|---|---|---|---|---|")
        for role, m in dev["models"].items():
            u = dev["evaluation"][role]["untuned_uncalibrated_metrics"]
            s = dev["evaluation"][role]["uncalibrated_metrics"]
            w(f"| {role} | {_f(u['roc_auc'])} | {_f(u['log_loss'])} | {_f(s['roc_auc'])} | {_f(s['log_loss'])} | "
              f"{m['search_seconds']} | {m['n_configurations']} |")
        w("")
        b = dev["paired_bootstrap_tuning"]
        w(f"Paired bootstrap (tuning; {b['n_used']} used, {b['n_failed']} failed resamples), challenger − baseline:")
        w("")
        w("| metric | baseline | challenger | difference | 95% CI |")
        w("|---|---|---|---|---|")
        for k, v in b["metrics"].items():
            w(f"| {k} | {_f(v['baseline'])} | {_f(v['challenger'])} | "
              f"{_f(v['difference_challenger_minus_baseline'])} | {_ci(v['difference_ci'])} |")
        w("")
        rs = dev["repeated_cv"]["summary"]
        w(f"Repeated development CV ({dev['repeated_cv']['scheme']}): AUC difference mean "
          f"{_f(rs['auc_difference']['mean'])} (sd {_f(rs['auc_difference']['sd'])}); challenger better on log loss "
          f"in {rs['folds_challenger_better_log_loss']}/{len(dev['repeated_cv']['folds'])} folds.")
        w("")
    if sel:
        w("## Frozen selection")
        w(f"- Rule: {sel['rule']}")
        w(f"- Log-loss difference CI: {_ci(sel['log_loss_difference_ci'])} → champion **{sel['champion']}** "
          f"({sel['champion_family']})")
        w(f"- Frozen at {sel['frozen_utc']}; threshold {sel['threshold']} (illustrative)")
        w("")
    if fin:
        w("## Final holdout audit (evaluated once)")
        w(f"Accessed by {fin['accessed_by']} at {fin['accessed_utc']}; n={fin['n']}, prevalence {fin['prevalence']:.4f}")
        w("")
        w("| model | metric | tuning | final |")
        w("|---|---|---|---|")
        for role, mets in fin["development_vs_final"].items():
            for k, v in mets.items():
                w(f"| {role} | {k} | {_f(v['tuning'])} | {_f(v['final'])} |")
        w("")
        w("| metric | difference (challenger − baseline) | 95% paired CI |")
        w("|---|---|---|")
        for k, v in fin["paired_bootstrap_final"]["metrics"].items():
            w(f"| {k} | {_f(v['difference_challenger_minus_baseline'])} | {_ci(v['difference_ci'])} |")
        w("")
        for role, ev in fin["evaluation"].items():
            cm = ev["metrics"]["confusion_matrix"]
            w(f"- {role} @ {ev['metrics']['threshold']}: precision {_f(ev['metrics']['precision'])}, recall "
              f"{_f(ev['metrics']['recall'])}, specificity {_f(ev['metrics']['specificity'])}, CM {cm}")
        w("")
    else:
        w("## Final holdout audit\nNot run.\n")
    st = evidence.get("stress")
    if st:
        w("## Stress suite (tuning partition)")
        w(st["caveat"])
        w("")
        roles = list(st["clean"])
        w("| scenario | kind | " + " | ".join(f"{r} ΔAUC | {r} Δlog loss" for r in roles) + " |")
        w("|---|---|" + "---|---|" * len(roles))
        for s in st["scenarios"]:
            w(f"| {s['id']} | {s['kind']} | " + " | ".join(
                f"{s['models'][r]['delta_roc_auc']:+.4f} | {s['models'][r]['delta_log_loss']:+.4f}" for r in roles) + " |")
        w("")
    ex = evidence.get("explanations")
    if ex:
        w("## Explanations (SHAP, development sample)")
        for role, m in ex["models"].items():
            top = ", ".join(f"{g['feature']} ({g['mean_abs_shap']:.3f})" for g in m["global"][:6])
            w(f"- {role}: {m['algorithm']}; additivity ok={m['additivity_ok']} "
              f"(max err {m['additivity_max_abs_error_calibrated']:.1e}); top mean|SHAP|: {top}")
        for lim in ex["limitations"]:
            w(f"  - {lim}")
        w("")
    w("## Deterministic checks (flags for a human; never decisions)")
    for c in checks:
        w(f"- [{c['status']}] {c['check_id']}: {c['message']}")
    w("")
    w("## Findings")
    if not findings:
        w("None recorded.")
    for f in findings:
        w(f"- #{f['finding_id']} [{f['severity']}/{f['category']}] **{f['title']}** — {f['status']}; owner "
          f"{f['owner']}; evidence `{f['evidence_ref']}`. {f['description']}")
    w("")
    w("## Review events (append-only; actor names are self-entered, not authenticated)")
    w(f"Hash chain intact: {chain['ok']} ({chain['n_events']} events). A local SQLite file is not tamper-proof.")
    w("")
    for e in events:
        w(f"- {e['at_utc']} {e['actor']} ({e['actor_role']}): {e['event_type']} {e['from_state']} → {e['to_state']}")
    w("")
    w("## Limitations")
    w("- Cross-sectional 2005 Taiwan sample; no out-of-time or real deployment-drift evidence.")
    w("- Bootstrap intervals are conditional on the fitted models and evaluation sample.")
    w("- Slice results are descriptive; not a legal fairness assessment.")
    w("- Stress and drift batches are simulations derived from the same historical sample.")
    w("- No expected-credit-loss, IFRS 9 or capital conclusions can be drawn from this dataset.")
    return "\n".join(L) + "\n"
