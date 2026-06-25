# SensorFlow Operator-First Search — Build Recipe (paste-by-paste)

Paste CHUNK 0 first to set context. Then paste CHUNK 1 → 8 one at a time, verifying each before the
next. Each step assumes the agent still has CHUNK 0 in the session. (Full rationale lives in the
design doc + spec; this file is only what you paste.)

---

## CHUNK 0 — Context (paste first, don't implement anything)

You are working on **SensorFlow**, a lightweight loop that improves an ML candidate (a Python file
doing feature selection + decision-tree training) via LLM-driven mutation. The iterative loop
**already exists**: each try picks a parent from the Pareto frontier, mutates it with feedback,
evaluates, and records the result in a flat in-memory archive. No acceptance gate — every result
(including failures) is recorded; the frontier is the implicit quality filter.

**We are adding operator-first search.** Instead of one generic "pick frontier parent → generic
feedback → mutate" step, each try first picks a **move type** — `repair`, `improve`, or `combine` —
and tailors *which parent(s)* and *what feedback* to that move. This is an additive layer on top of
the existing loop.

**What already exists (rely on it, don't rebuild):**
- The loop, the archive, `select_parent(successful, metric_policy) -> entry` (random over the
  frontier, module RNG seeded 42), and `summarize_feedback(archive, metric_policy, parent_try_id)
  -> str` (the current 3-block feedback string).
- The evaluator catches candidate failures → `result` dict with `status="error"`, `metrics=None`,
  `failure_reason`. Metrics on success include four diagnostic `nb_segments_*` confusion-matrix
  counts (NOT in `metric_policy`).

**Key facts (don't re-derive):**
- Archive entry = `{"try_id": int, "candidate": <Path>, "result": <EvaluationResult>,
  "parent_try_id": int | None}` (also mirrors `status`, `accuracy`, … as top-level row fields).
  **`entry["result"]` is an `EvaluationResult` OBJECT, not a dict** (from `evaluator.py`): read its
  fields by **attribute** — `entry["result"].status`, `entry["result"].metrics`,
  `entry["result"].failure_reason`. (`.status` is `"ok"`/`"error"`; `.metrics` is a dict or `None`.)
  For the failure filter you may use the mirrored top-level `entry["status"]` (that's how the
  existing `successful` list is built). Confirm both in CHUNK 1.
- `metric_policy`: `dict[str,str]`, values `"max"`/`"min"`, exact keys: `accuracy` (max),
  `f1_score_class1` (max), `depth` (min), `num_leaf_nodes` (min). Do not rename these.
- `pareto_frontier_try_ids(rows, metric_policy) -> list[int]` in `get_pareto.py`; `rows =
  list[(try_id:int, metrics:dict)]`. Returns try_ids, not entries. Do not modify it.
- Candidate source path: `tries/try_{id:03d}/candidate.py` (also in `entry["candidate"]`).
- Module RNG seeded 42 (the same one `select_parent` uses) — use it for all randomness; never bare `random.*`.

**Locked design decisions (don't relitigate):**
- Three operations only: `repair`, `improve`, `combine`.
- Operation choice: if a recent failure exists, fire `repair` with **probability 0.3**; else if the
  frontier has **≥ 3** points, `combine`; else `improve`.
- `improve` parent selection stays **random over the frontier** (the existing `select_parent`).
- `repair` works on **failures within the last 5 tries**, picks one at random.
- `combine` picks **two parents = best-on-two-random-metrics**.
- `improve` target metric **cycles** the `metric_policy` keys (all of them, incl. `accuracy`).
- No acceptance gate; failures still recorded; the frontier stays the filter.

**Global rules for every step:**
- **Additive/minimal.** Do NOT refactor data layers or introduce new core objects. Work with the
  existing archive **dicts** — there is no `Candidate` class; do not add one.
- **Feasibility first.** Before writing code in a step, confirm it against the real code and report
  any mismatch with what the step claims (paths, signatures, the `result` nesting, what
  `summarize_feedback`/`select_parent` currently do). If anything differs, **STOP and ask** — do not
  silently adapt.
- You may decide local details (variable names, number formatting). You may NOT change contracts,
  control flow, or any "DON'T".
- **Robustness is a non-goal.** Anything that can't happen in a correct run just raises naturally —
  no validators, no try/except (except where a step explicitly says so).
- When ambiguous, ask one clarifying question rather than guessing.

Acknowledge you've read this and wait for CHUNK 1. Do not write code yet.

---

## CHUNK 1 — Repair trigger: recent failures + probabilistic gate

**Model:** Haiku.

**Goal:** decide whether this try should be a repair, based on recent failures.

**Do:**
- `recent_failures(archive) -> list[dict]`: return the entries among the **last 5** of `archive`
  whose status is not ok. Use the mirrored top-level `entry["status"] != "ok"` (that's how the
  existing `successful` list is built); equivalently `entry["result"].status != "ok"`. Empty list if none.
- `should_repair(archive) -> bool`: `fails = recent_failures(archive)`; return
  `bool(fails) and _rng.random() < 0.3`. The `bool(fails)` must short-circuit **before** the draw.

**Don't:**
- Don't scan the whole archive — only the last 5 entries.
- Don't draw from `_rng` when there are no recent failures (it perturbs reproducibility).
- Don't add a cooldown, counter, or rate logic — probability only.

**Verify:** with no recent failure, `should_repair` is `False` and never calls `_rng`; with a recent
failure, it's `True` ~30% over many seeded draws; `recent_failures` returns only non-ok among the last 5.

**Feasibility first:** confirm the archive is a list of dicts with `entry["result"]["status"]`, and
report the name of the shared seeded RNG (the one `select_parent` uses). Report before coding.

---

## CHUNK 2 — `choose_operation(archive, frontier_ids)`

**Model:** Haiku.

**Goal:** the operation gate — return `"repair" | "improve" | "combine"`.

**Do:**
- `choose_operation(archive, frontier_ids: list[int]) -> str`:
  `if should_repair(archive): return "repair"`; `elif len(frontier_ids) >= 3: return "combine"`;
  `else: return "improve"`.
- `frontier_ids` is whatever `pareto_frontier_try_ids(...)` already returns at the call site.

**Don't:**
- Don't recompute the frontier inside this function — take the already-computed `frontier_ids`.
- Don't add weights, schedules, or a 4th branch.

**Verify:** crafted inputs hit each branch — recent failure → repair; no failure + ≥3 frontier →
combine; no failure + <3 frontier → improve.

**Feasibility first:** confirm where in the loop `pareto_frontier_try_ids` is already called so
`frontier_ids` is available to pass in. Report the call site.

---

## CHUNK 3 — `choose_target_metric(improve_count)`

**Model:** Haiku.

**Goal:** for an `improve` try, pick which metric to push, by cycling the policy keys.

**Do:**
- `choose_target_metric(improve_count: int) -> tuple[str, str]`:
  `keys = list(metric_policy)`; `k = keys[improve_count % len(keys)]`; `return (k, metric_policy[k])`.
- `improve_count` = how many improve tries have happened so far (caller supplies it; see CHUNK 7).

**Don't:**
- Don't exclude any key — cycle all of them, including `accuracy`.
- Don't pick "weakest metric" or normalize anything — pure round-robin.

**Verify:** `choose_target_metric(0,1,2,3,4)` cycles the keys in policy order and wraps at the end.

**Feasibility first:** confirm `metric_policy` is importable here and report its key order. Report.

---

## CHUNK 4 — Combine picker: `best_on_metric` + `pick_frontier_extremes`

**Model:** Haiku.

**Goal:** for a `combine` try, pick two complementary frontier parents.

**Do:**
- `best_on_metric(metric, frontier_entries) -> entry`: return the frontier entry whose
  `entry["result"]["metrics"][metric]` is highest (if `metric_policy[metric]=="max"`) or lowest
  (`"min"`). Natural first-seen tie-break.
- `pick_frontier_extremes(frontier_entries) -> ((entry_a, m1), (entry_b, m2))`:
  `m1, m2 = _rng.sample(list(metric_policy), 2)`; `a = best_on_metric(m1, …)`,
  `b = best_on_metric(m2, …)`; if `a is b`, set `a, b = _rng.sample(frontier_entries, 2)`; return the
  two (entry, metric-name) pairs. The metric names are the "strong" labels the combine prompt uses.

**Don't:**
- Don't loop/re-draw on collision — one fallback to two random distinct entries, then return.
- Don't normalize across metrics or compute a "spread."

**Verify:** returns two distinct entries plus two metric names; on a frontier where one entry is best
on all metrics, the fallback still returns two distinct entries.

**Feasibility first:** confirm you can get frontier **entries** (not just ids) at the call site —
i.e. there's a try_id→entry lookup available. Report how entries are looked up by try_id.

---

## CHUNK 5 — Operation-aware parent selection

**Model:** Haiku.

**Goal:** route parent selection by operation, reusing the existing `select_parent` for improve.

**Do:**
- `select_parents(op, archive, successful, metric_policy) -> list[dict]` (returns archive entries):
  - `"improve"` → `[select_parent(successful, metric_policy)]` (the EXISTING function, unchanged).
  - `"repair"` → `[_rng.choice(recent_failures(archive))]`.
  - `"combine"` → handled by the caller via `pick_frontier_extremes` (this function may raise or be
    skipped for combine — see CHUNK 7; do NOT duplicate the extremes logic here).
- Always returns a list (one entry for improve/repair).

**Don't:**
- Don't reimplement frontier-random selection — call the existing `select_parent`.
- Don't put combine's two-parent logic here (it needs metric labels; the caller owns it).

**Verify:** improve returns one frontier entry via `select_parent`; repair returns one recent-failure
entry; both as 1-element lists.

**Feasibility first:** confirm the exact current signature of `select_parent` and what `successful`
is (the list of ok entries) at the call site. Report.

---

## CHUNK 6 — Operation-specific feedback builders

**Model:** **Sonnet** — verbatim prompt text + must reuse `summarize_feedback`'s renderer + `EvaluationResult` attribute access (easy to silently get wrong on Haiku).

**Goal:** three feedback strings, one per operation, replacing the single generic feedback for these
moves. Render each body **verbatim**, substituting only the `{...}` fields. Read candidate source
from the entry's path with `open(entry["candidate"]).read()`.

**Do — implement three builders returning the exact text below:**

`build_repair_feedback(fail_entry) -> str`:
```
TASK: The candidate below failed when it was run. Debug it and return
a working version.

Failed candidate (try {try_id}):
{candidate source}

What went wrong:
{result["failure_reason"]}
```

`build_improve_feedback(parent_entry, target_metric, direction, frontier_entries) -> str`:
```
TASK: Improve the candidate below on one target metric. Push that
metric as far as you can. Try to hold the other scores where they are,
but you may sacrifice them if it meaningfully helps the target.

Target this run: {target_metric} → {direction}

Current candidate (try {try_id}):
{candidate source}
Current scores: {scores of parent}

Other candidates on the current frontier:
  try {id}: {scores}      ← one line per frontier entry, capped at NEIGHBOR_CAP = 5

GOALS, in order:
1. Move {target_metric} in the {direction} direction as much as you can.
2. Prefer to keep the other scores steady — but trading some away for
   a real gain on the target is allowed.
3. Make one clear, deliberate change, not a scattershot rewrite.
```

`build_combine_feedback(entry_a, strong_a, entry_b, strong_b) -> str`:
```
TASK: Two candidates below each do something well. Think about whether
their strengths can be combined, and if so, build ONE new candidate
that captures both. This is recombination of ideas — don't blindly
average or paste the two together.

Candidate A (try {a_id}) — strong on {strong_a}:
{A source}
A scores: {A scores}

Candidate B (try {b_id}) — strong on {strong_b}:
{B source}
B scores: {B scores}

GOALS, in order:
1. Reason about why each candidate is good, and whether their
   strengths are actually combinable.
2. If they are, build one candidate that keeps A's strength on
   {strong_a} and B's on {strong_b} as much as possible.
   A middle-ground tradeoff between them is fine.
3. If they genuinely aren't combinable, take the more promising
   approach and improve it rather than forcing a bad merge.
4. Produce one coherent file, not a stitched-together hybrid.
```

- For "scores", reuse `summarize_feedback`'s existing renderer (it iterates `metric_policy.keys()`
  and formats `f"{key}={value:.4f}"`); read values from `entry["result"].metrics` (attribute, a
  dict keyed by metric name). Show only `metric_policy` keys, not the `nb_segments_*` diagnostics.
- `failure_reason` is `fail_entry["result"].failure_reason` (attribute, not subscript).

**Don't:**
- Don't reword the prompt text. Don't add a baseline line to improve. Don't show scores in repair.
- Don't show the `nb_segments_*` counts in any block.
- `NEIGHBOR_CAP = 5` is a placeholder — flag it for owner confirmation, don't treat it as final.

**Verify:** each builder's output matches the body above with fields slotted; repair has no scores;
improve shows ≤5 frontier rows and no baseline; combine shows both sources + strong labels.

**Feasibility first:** report exactly how `summarize_feedback` currently renders scores (so you reuse
that format), and confirm `open(entry["candidate"]).read()` gives the candidate source. Report.

---

## CHUNK 7 — Wire operations into the loop body

**Model:** **Sonnet** — only chunk editing live control flow; must swap `feedback=` ONLY and leave `prompt=`/`_build_prompt` untouched (over-reach risk on Haiku).

**Goal:** replace the loop's current "select parent → summarize_feedback → mutate" body with the
operation-first flow. Additive edit to the existing per-try body only.

**Do (in the loop body, per try):**
1. Build `frontier_ids` the way the loop already does (`pareto_frontier_try_ids(rows, metric_policy)`).
2. `op = choose_operation(archive, frontier_ids)`.
3. Pick parent(s) + build feedback per op:
   - `repair` → `p = select_parents("repair", …)[0]`; `feedback = build_repair_feedback(p)`;
     mutate base = `p`.
   - `improve` → `p = select_parents("improve", …)[0]`;
     `feedback = build_improve_feedback(p, *choose_target_metric(improve_count), frontier_entries)`;
     mutate base = `p`.
   - `combine` → `(a, sa), (b, sb) = pick_frontier_extremes(frontier_entries)`;
     `feedback = build_combine_feedback(a, sa, b, sb)`; mutate base = `a`.
4. Pass the operator `feedback` string into the existing mutate call as the **`feedback=` argument**
   (the slot `summarize_feedback`'s output currently fills, at `runner.py:477`/`491`). **Leave
   `prompt=` and `_build_prompt(...)` untouched** — `mutate(parent, prompt, output_dir, feedback)`
   takes both; you are only swapping what goes into `feedback=`. Pass the mutate base's candidate
   path as `parent` (one parent, as today).
5. `improve_count` = number of prior `improve` tries — derive from the per-try log/archive (see
   CHUNK 8); flag this choice for owner confirmation.

**Don't:**
- Don't change eval, record, or the frontier computation.
- Don't touch `prompt=` / `_build_prompt` — operators only replace the `feedback=` string.
- Don't pass two parents to mutate — combine's intent lives in the feedback; base = A.
- Don't keep calling `summarize_feedback` for these ops (improve's builder replaces it; it may
  internally reuse summarize_feedback's renderer, but the loop no longer calls it directly).

**Verify:** a short run shows all three ops occurring; failed candidates still recorded and reused by
repair; eval/record/visibility unchanged.

**Feasibility first:** report the current loop-body lines that call `select_parent`,
`summarize_feedback`, and `mutate`, and confirm `frontier_entries` (id→entry) is obtainable there. STOP if the mutate signature differs from "one parent path + feedback string".

---

## CHUNK 8 — Log the operation

**Model:** Haiku.

**Goal:** record which operation fired each try, for analysis.

**Do:**
- Add two columns to the existing `tries.csv` writer: `operation` (the op string) and
  `parent_try_ids` (the chosen parents' try_ids, comma-joined — one for repair/improve, two for combine).
- This is also the source for `improve_count` in CHUNK 7 (count rows where `operation == "improve"`).

**Don't:**
- Don't add a separate log file or reorder existing columns.
- Don't log a "reason" string — just op + parent ids.

**Verify:** each `tries.csv` row gains `operation` + `parent_try_ids`; existing columns unchanged.

**Feasibility first:** report the current `tries.csv` column list and where the row is written. Report.

---

### Order recap
`0 context → 1 repair trigger → 2 choose_operation → 3 target cycle → 4 combine picker →
5 parent routing → 6 feedback builders → 7 wire into loop → 8 log op`. 1–6 are independent leaves
(any order); 7 integrates them; 8 is independent of 7 but feeds it `improve_count`.

**Model split:** Haiku for 0–5 and 8; **Sonnet for 6 (verbatim prompts + renderer reuse + attribute
access) and 7 (live control-flow edit, feedback-only swap).** Bias is Haiku; the two Sonnet chunks
are the ones where a silent Haiku slip would cost debugging time later.



Got it — keep the prints in, no `# SMOKE` markers, no cleanup step. Here's the trimmed brief:

---

**Goal: verify the operator layer is live and correct via two smoke runs, with evidence in stdout and `tries.csv`. Add brief permanent print lines for observability.**

**Step 1 — add observability prints.**
In the per-try loop, after `op = choose_operation(...)` and parent selection, add one concise line:
```
print(f"[op] try={try_id} op={op} parents={[p['try_id'] for p in chosen]} "
      f"target={target_metric if op=='improve' else '-'} frontier={len(frontier_ids)} fails={len(recent_failures(archive))}")
```
Match the real in-scope variable names.

**Step 2 — Smoke A: framework integrity (no LLM).** Run `--no-mutate` (or the parent-copy path) for **5 tries**. Confirm: completes without exception; `[op]` prints every try with `op` in {repair, improve, combine}; improve `target=` cycles (not stuck); `tries.csv` has populated `operation` + `parent_try_ids` columns, existing columns unchanged; no error around `entry["result"].status`/`.metrics`. Paste the 5 `[op]` lines + the new CSV columns.

**Step 3 — Smoke B: real run, 8–10 tries (LLM on).** Confirm: completes without unhandled exceptions; failures recorded with `status="error"` (don't crash the loop); all three ops appear in the `operation` column (if frontier never hits 3, combine may not fire — say so explicitly, don't call it a pass); at least one improve target differs from a prior improve's; combine rows show two distinct parent ids, repair/improve one. Paste the `[op]` lines, the CSV columns, and a one-line per-op tally.

**Step 4 — verdict.** State plainly whether the layer is (a) running and (b) selecting correctly, citing the evidence. If anything looks wrong, report and stop — don't tweak operator logic to force a pass.

---

Same two watch-points when it reports: improve `target=` actually cycling (stuck-on-`accuracy` = `improve_count` not advancing), and failures landing in the archive without crashing.




Two fixes. Do not add a preamble-stripper or any code-extraction logic — repair will handle malformed candidates itself. Read-only confirm each feasibility point before editing; if reality differs from what's stated, STOP and report.
Fix 1 — per-try failure isolation (stop the crash).

In the per-try loop body of run_mutation_search (the loop at ~runner.py:1019, body ~458), wrap the work done for a single try — selection, feedback build, mutate, evaluate, record — in a try/except Exception as e. On exception: print one line (print(f"[try {try_id}] skipped: {e}")), and continue to the next try. One bad try must never kill the run.

Don't wrap the whole loop — wrap the body, so only the failing try is skipped.
Don't swallow silently — print the try id + error.
Feasibility: report the current loop-body structure (control flow only) and confirm there's no existing try/except already there.

Fix 2 — failure entries carry their candidate path, so repair can fix them.

Currently a mutation-phase failure records an archive entry with candidate = None, so build_repair_feedback's open(fail_entry["candidate"]) throws. The intent: when a candidate file was written to disk (even if it later failed validation — e.g. prose preamble + syntax error), the failure entry should store that path, so repair can open the malformed source and fix it.

Find where the failure/error archive entry is constructed (compare to the success-path entry construction). Report whether candidate is set on the failure path or left None.
Find where the candidate .py is written to disk relative to where validation throws. Report: on a validation failure, does the file exist on disk? (From try 6, it does — try_006/candidate.py exists with the prose in it.)
If the file exists, set the failure entry's candidate to that path (same value the success path would use). Do not change the success path.
Feasibility: report the file:line of both the failure-entry construction and the candidate-write, and confirm the written path is in scope at the entry construction.

After both fixes, repair works end-to-end: a malformed candidate (preamble, bad char, etc.) is recorded with its path → recent_failures surfaces it → build_repair_feedback opens the file → the LLM sees the source + the failure_reason and fixes it (e.g. strips the narration). No filter, no stripper needed.
Verify: re-run 8–10 tries. Confirm: (1) if any try throws, you see the [try N] skipped line and the run continues to completion; (2) a failed candidate's entry has a non-null candidate path; (3) when repair fires on such an entry, it builds feedback without TypeError and produces a fixed candidate. Report the [op] lines and any skip lines.
Don't: don't add code-extraction/preamble-stripping anywhere; don't filter failures out of the repair pool (repair is meant to fix them); don't touch the success path or the mutation engine's output handling.
