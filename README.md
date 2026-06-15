# portfolio-draft

# SensorFlow Iterative Loop — Decision Log

## Agreed Behavior

Each try, the loop picks its parent via `random.choice(get_pareto(successful archive entries))`, calls the existing mutate with that parent’s source plus a new feedback section built by `summarize_feedback(archive, metric_policy)`, evals, and appends the result (including failures) to the flat archive — no acceptance gate, no `improved()`, Pareto computed from the archive exactly as today, with an optional `parent_try_id` field added per entry.

## Feedback Content (v1)

`summarize_feedback(archive, metric_policy)` returns a pre-rendered string (delivery: pre-rendered string passed into existing `mutate` signature — no data-layer change). Rendered as compact labeled lines (not JSON). Three blocks:

1. **Anchor** — baseline metrics + best-so-far per policy metric.
1. **Frontier** — Pareto frontier members as metric rows, the parent being mutated marked inline (`← mutating this`), capped at top N rows with “+K more” overflow note. Single block; no separate parent block (parent is on the frontier, avoid redundancy).
1. **Confusion matrix** — for the parent only. Surfaces the class imbalance (rare class = car crash): caught-rate and false-alarm-rate with raw denominators.

Confusion matrix capture: add 4 raw counts (tn, fp, fn, tp) to the metrics dict, evaluator-side, from the matrix already computed for the existing PNG. **Diagnostic only — these counts must NOT go into `metric_policy`** (they are reported, not Pareto objectives; f1_score_class1 remains the rare-class policy objective).

Deferred from v1: chosen-vs-fitted cap surfacing (needs new capture path); per-direction history / anti-flip-flop (needs change-labels).

## Failure Contract

- **Where:** the evaluator wraps ONLY the candidate train+eval execution span in try/except. Data loading, metric computation on success, and everything outside candidate execution still crash loudly (those are framework bugs, not candidate failures).
- **On catch:** print one diagnostic line (e.g. `[eval] candidate failed: <first line of error>`) for live visibility, AND return an error dict so the loop continues.
- **Error dict shape:** identical shape to success, with `metrics: {}` (empty dict, never None, never absent), `status: "error"`, `failure_reason: str(e)` (the exception message, not a traceback). On success: `status: "ok"`, `failure_reason: None`.
- **`status` values:** exactly two — `"ok"` and `"error"`.

## Success Predicate

- An archive entry is “successful” iff `status == "ok"`. Single source of truth; never inspect metrics to determine success.
- Applied as a runner-side filter producing the `successful` list:
  `successful = [e for e in archive if e.status == "ok"]`
- This `successful` list feeds BOTH parent selection and frontier computation.
- `get_pareto` is NOT modified — filtering happens at the call site in the runner, before `get_pareto` is called.

## Failure Contract

- **Where:** the evaluator wraps ONLY the candidate train+eval execution span in try/except. Data loading, metric computation on success, and everything outside candidate execution still crash loudly (those are framework bugs, not candidate failures).
- **On catch:** print one diagnostic line (e.g. `[eval] candidate failed: <first line of error>`) for live visibility, AND return an error dict so the loop continues.
- **Error dict shape:** identical shape to success, with `metrics` = `{}` (empty dict, never None, never absent), `status` = `"error"`, `failure_reason` = `str(exception)` (the message, not a full traceback). On success: `status` = `"ok"`, `failure_reason` = `None`.
- **Status values:** exactly two — `"ok"` and `"error"`.

## Success Predicate

- A single predicate everywhere “usable” is needed: **`status == "ok"`**. Metrics are never inspected to decide success.
- Applied as a **runner-side filter** producing the `successful` list, which feeds BOTH parent selection and frontier computation:
  `successful = [e for e in archive if status == "ok"]`
  `parent = random.choice(get_pareto(successful, metric_policy))`
- `get_pareto` is **not modified** — it continues to assume all inputs have metrics. Filtering happens at the call site, in the (new) runner code.

## Archive (net-new — does not exist yet)

**Current reality:** there is no archive today. A run only produces per-try `evaluation_result.json` objects written to `tries/try_NNN/.../`. There is no in-memory accumulation and no runtime Pareto over a live collection. The archive must be INTRODUCED — it is additive, not a refactor of existing data.

**Decision (per “boundary-only persistence” principle):**

- Archive = an **in-memory list of result entries**, built during the run (`results = []` in the runner, appended each iteration).
- Per-try `evaluation_result.json` write remains the persistence boundary.
- **No reading JSON back mid-loop** — parent selection and feedback read the in-memory list, not the disk files.
- Entry shape: since it’s net-new, use a **dict** (or small dataclass) — no legacy tuple unpacks to break, and future fields (generation, niche, score…) add cleanly. Each entry carries at least: `try_id`, `candidate`, `result`, `parent_try_id`.
- `parent_try_id`: set by the runner at append time (not the evaluator). Baseline seed = `None` (it is the root). Implicit tree — lineage is reconstructable from the pointers; no explicit tree/graph object is built.

**Open / to confirm:** whether per-try JSON is already written by current code, or only happens because tries are run one-at-a-time by hand right now.
