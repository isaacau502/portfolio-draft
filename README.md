# Visibility Suite — Coding-Agent Instruction Set

## Background (read first — this is a fresh session)

**What SensorFlow is.** SensorFlow is an LLM-driven program-evolution loop. A *runner* improves an ML candidate script (feature selection + decision-tree training on a sensor classification task) over many iterations. Each iteration — a **"try"** — does: select a parent from the current Pareto frontier → **mutate** it into a new candidate `.py` via an LLM call (Amazon Bedrock) → **evaluate** the candidate (trains/evaluates the model, returns a metrics dict + a status) → append the result to an in-memory **archive** (a list of per-try entries). Optimization is **multi-objective** over a `metric_policy` (e.g. accuracy = maximize, node-count = minimize, depth = minimize); "best" is the Pareto frontier, not a single score. The seed/baseline is **`try_000`** — a normal archive entry with `parent_try_id = None`, evaluated once before the loop starts. A `--no-mutate` mode runs the same loop but *copies* the parent instead of calling the LLM (a cheap framework smoke-test). Output lands under `runs/<run>/`, with per-try directories and a `summary.json`.

**Why we're building this suite.** The loop runs, but right now it doesn't beat the baseline on any objective. We don't yet know whether that's because the *machinery* can't produce improvement, or because the *baseline is already maxed out* (no headroom left to find). To settle it, we'll run the loop from a deliberately **crippled baseline** and watch whether it climbs back toward the real baseline's numbers — climbs → machinery works, real baseline was just saturated; stays flat despite obvious headroom → the machinery is the problem. **This suite is the instrumentation that lets us read that diagnostic.** Everything below exists to make a climb (or its absence) visible: a per-try table, per-try timing, a success/failure tally, and a Pareto plot that shows the frontier moving over iterations toward a target. That purpose is *why* the plot choices later (color points by try index, "outwards = better", mark the seed and a baseline-target reference) are what they are — they make movement legible at a glance. Build for that purpose; add nothing beyond it.

---

## What you're building

Build a thin visibility suite for SensorFlow runs. Every design decision below is already made — **do not invent alternatives, do not refactor beyond what's specified, do not rename anything.** If something can't be done as written, stop and report rather than improvising.

Scope is four components, built in dependency order: timing capture → CSV table → Pareto plot → run tally. (A best-so-far line plot was considered and deliberately cut — do **not** build it.)

## Model ratings

| Chunk | Task | Model | Why |
|---|---|---|---|
| 0 | Orientation / locate anchors | **Haiku** | Pure read + grep; no edits. |
| 1 | Timing instrumentation | **Haiku** | Two `perf_counter` wraps; mechanical. |
| 2 | `tries.csv` | **Haiku** | `DictWriter` + append; fully specified. |
| 3 | Pareto plot upgrade | **Haiku ok, Sonnet if you want it right first-pass** | Matplotlib (axis inversion + frontier line + colorbar) is easy to get subtly wrong. Haiku can do it from this spec, but eyeball the PNG. |
| 4 | Success/fail tally | **Haiku** | Two counts + JSON write; trivial. |

Strong Haiku lean overall. Chunk 3 is the only place a model bump buys you fewer iterations.

---

## Chunk 0 — Orientation (Haiku, no edits)

Locate and report each, with `file:line`. Later chunks depend on these being exact. Make no edits.

1. **Per-try archive-append site** — where each try's entry (with `try_id`, `parent_try_id`, `status`, metrics/result) is built and appended to the in-memory archive. Report the function name and whether a `record(...)`-style helper exists or it's inlined in the loop.
2. **Runner call sites** for `evaluate(candidate)` and the mutation/LLM call (`mutate(parent, …)`), each `file:line`.
3. **Run-directory init** — where `runs/<run>/` is created at run start (a place to write a CSV header exactly once).
4. **End-of-run summary writer** — where `summary.json` is written.
5. **Pareto plotting** — the existing `plot_pareto` function (expected in `get_pareto.py`) and `pareto_frontier_try_ids`; report `file:line` and what `plot_pareto` currently draws and where it saves.
6. **Metric names** — the actual keys in `metric_policy` and in the evaluator's returned metrics dict. Report the current names for accuracy, node-count, and depth. (Note: a rename of `max_depth`/`max_leaf_nodes` is a *separate, out-of-scope* task — every chunk below uses whatever names exist **now**.)
7. **Token usage** — whether the Bedrock response object exposes token counts, and the access path if so.

Output: `path:line — note` per item. No edits.

---

## Chunk 1 — Timing instrumentation (Haiku) — depends on Chunk 0

In the runner loop, wrap the two call sites with `time.perf_counter()` (not `time.time()`):

```python
t0 = time.perf_counter()
result = evaluate(candidate)
eval_time_s = time.perf_counter() - t0
```

and identically around the mutate/LLM call → `llm_time_s`.

Add three fields to the per-try archive entry (at the append site from Chunk 0):
- `eval_time_s` — float, always present.
- `llm_time_s` — float; **`None` for try_000 (baseline) and any try with no LLM call** (e.g. the `--no-mutate` copy step).
- `tokens` — total token count from the Bedrock response if Chunk 0 found it exposed; otherwise `None`.

Constraints: measure **only at the runner call sites** — do not measure inside the evaluator or mutation engine, and do not change their return contracts.

Files: runner only.
Verify: every entry carries `eval_time_s` (float), `llm_time_s` (float or `None`), `tokens` (int or `None`).

---

## Chunk 2 — `tries.csv` (Haiku) — depends on Chunk 1

Write one flat CSV, one row per try, appended live.

- **Path:** `runs/<run>/tries.csv`.
- **Columns, in this order:** `try_id, parent_try_id, status`, then one column per `metric_policy` key in policy order (using the current metric names from Chunk 0), then `eval_time_s, llm_time_s, tokens, failure_reason`.
- **Header:** written exactly once, at run-directory init (Chunk 0 #3).
- **Writer:** `csv.DictWriter` with `restval=''` so failed tries (metrics `None`) leave their metric cells blank.
- **Per-try row** (at the append site): pull metric values from the entry's metrics dict by policy key (missing → blank via `restval`); include `status`, the three timing/token fields, and a **one-line** `failure_reason` (first line only — the full traceback stays in the per-try JSON; do not duplicate it here).
- **Do not** write confusion-matrix raw counts or any nested fields into the CSV.
- **Do not** rename any metric key.

Files: runner (and the run-init location for the header).
Verify: header written once; a successful row has metric values; a failed row has blank metric cells + a one-line reason; the file is appended per try (not rewritten); `pandas.read_csv` parses it cleanly.

---

## Chunk 3 — Pareto plot upgrade (Haiku ok / Sonnet recommended) — depends on Chunk 2

Upgrade the existing `plot_pareto` (Chunk 0 #5) to read **from `tries.csv`** and render the diagnostic view. All choices are fixed:

- **Axes:** accuracy on x, node-count on y (current metric names). **Do not plot depth.**
- **Points:** scatter all successful candidates (`status == "ok"`); skip failures (they have no coordinates).
- **Color:** by `try_id`, sequential colormap `viridis`, with a colorbar labeled **"try index"**.
- **"Outwards is better":** leave the accuracy axis normal (higher = right); **invert the node-count axis** (`ax.invert_yaxis()`) so fewer nodes points outward. Good corner = top-right.
- **Frontier line:** draw the final Pareto frontier through the non-dominated successful points. Reuse `pareto_frontier_try_ids` for membership — **do not modify that function; this is display-only.**
- **Seed marker:** mark `try_id == 0` with a distinct marker (e.g. star), label "seed".
- **Baseline reference:** accept a parameter `baseline_ref=(accuracy, nodes)`; if provided, mark it with a distinct marker (e.g. ✕) labeled "baseline target"; if `None`, omit. This is the *real* baseline's known numbers — it is **not** in `tries.csv`, so it must come in as a function argument (have the caller pass it; a constant is fine).
- **Save** to `runs/<run>/pareto.png` (same path as before).
- Reads `tries.csv` only — does not touch the live archive or the frontier logic.

Files: wherever `plot_pareto` lives (likely `get_pareto.py`), plus the caller that now passes `baseline_ref`.
Verify: PNG renders; points colored by `try_id` with a "try index" colorbar; node axis inverted so fewer-nodes is outward; frontier line through the non-dominated points; seed and (if passed) baseline markers present; no failed candidates plotted.

---

## Chunk 4 — Success/fail tally (Haiku) — depends on Chunk 2

At end of run, count archive entries by status:
- `n_ok = count(status == "ok")`, `n_error = count(status != "ok")`, `n_tries = total`.
- Print a one-line summary to stdout.
- Add to `summary.json` (Chunk 0 #4): `n_tries`, `n_ok`, `n_error`, and `failure_reasons` (the list of one-line reasons for the failed tries). **No** failure-type classification — just the counts and the raw one-line reasons.

Files: the `summary.json` writer.
Verify: `summary.json` contains the counts and `n_ok + n_error == n_tries`.

---

## Global guardrails (apply to every chunk)

- Do not modify the evaluator or mutation-engine return contracts.
- Do not rename metric keys — the `max_depth`/`max_leaf_nodes` rename is a separate task, out of scope here.
- Plots read `tries.csv`, never the live archive — keep capture and view decoupled so plots can be regenerated without re-running.
- The axis flip in Chunk 3 is display-only; `pareto_frontier_try_ids` is untouched.
- The best-so-far line plot is intentionally excluded. Do not add it.
