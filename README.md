Here's the scout brief — paste it to your scouting agent. It gathers everything the build recipe's per-chunk feasibility checks need, in one read-only pass.

---

**Read-only reconnaissance of the SensorFlow codebase.** Do not modify anything — only read and report. The goal is to confirm the facts an operator-layer build will rely on, and flag any mismatch. For each item, report the file + function/line + a one-line answer; if something doesn't exist or differs from what's described, say so explicitly (a mismatch is the most useful thing you can find). Keep the whole report skimmable.

1. **Shared RNG** — find the module-level seeded RNG (expected `random.Random(42)`). Report its exact variable name and confirm `select_parent` uses it for the frontier pick.

2. **`select_parent`** — report its exact current signature and what it returns (an archive entry? a try_id? a path?). Report what argument represents the successful/frontier entries it picks from, and how that list is built at the call site.

3. **`summarize_feedback`** — report its exact signature, and **paste the format of its output**: the block structure and, specifically, **how it renders a metrics dict into text** (the operator feedback builders must reuse that exact score format). Note whether it already shows frontier rows and parent scores.

4. **Archive entry shape** — confirm a single entry is a dict with keys `try_id`, `candidate`, `result`, `parent_try_id`. Confirm `result` is the eval dict and that status/metrics are nested as `entry["result"]["status"]` and `entry["result"]["metrics"]` (NOT top-level). Report `try_id` type (int or string).

5. **Candidate source path** — confirm the candidate `.py` is at `tries/try_{id:03d}/candidate.py` and is also stored in `entry["candidate"]`. Confirm `open(entry["candidate"]).read()` yields the source.

6. **Pareto function** — confirm `pareto_frontier_try_ids(rows, metric_policy)` in `get_pareto.py` takes `rows = list[(try_id, metrics_dict)]` and returns a list of try_ids. Report where the loop currently calls it, and **whether a try_id→entry lookup is available at that point** (needed to get frontier *entries*, not just ids).

7. **`metric_policy`** — report where it's defined and its exact keys + `"max"`/`"min"` values (expected: `accuracy` max, `f1_score_class1` max, `depth` min, `num_leaf_nodes` min).

8. **Mutate call** — report the exact signature of the mutation function and the current call site in the loop. Confirm it takes one parent (path) plus a feedback/prompt string, and report exactly where `summarize_feedback`'s output is passed in (the operator feedback will replace it there).

9. **Loop body** — report the lines of the per-try loop that call `select_parent`, `summarize_feedback`, and the mutate function, so the operator wiring knows what it's replacing.

10. **`tries.csv`** — report the current column list and where/how a row is written (the operation log will add two columns).

11. **Confusion-matrix counts** — confirm successful-eval metrics include the four `nb_segments_*` keys, and that they are NOT in `metric_policy` (so the feedback builders can exclude them from displayed scores).

For each item, if the reality differs from the description above, STOP flagging it inline — list all mismatches together at the end so they can be reconciled into the recipe before any building starts.






# SensorFlow — Operator-First Search: Coding Agent Recipe





> **Purpose of this doc.** A build recipe for implementing the operator-first prompted-search
> refactor (see the companion design doc). It is written to be followed by a **lightweight coding
> model**: ultra-specific, decomposed into small units, with no room for interpretation. The goal
> is to prevent *design drift* — every choice the design doc settled is pinned here as an
> instruction, not a discussion.
>
> **How to use this doc.** Work top-down. Sections are ordered to be read in sequence: context →
> components → resolved flags → data model → conventions → function inventory → signatures. Do not
> invent behaviour not written here. If an instruction is ambiguous or missing, STOP and flag it
> rather than guessing — a wrong guess reintroduces the drift this recipe exists to prevent.

---

## 1. Task context

- **System:** SensorFlow — a minimal LLM-driven program-evolution loop. An ML "candidate" script
  is mutated by an LLM over many iterations; each mutation is evaluated; results are kept in a flat
  archive whose Pareto frontier is the quality filter.
- **What we're building:** the shift from a single generic mutation prompt to **operator-first
  prompted search** — the loop first selects *which kind of move* to make (`repair` / `improve` /
  `combine`), then assembles move-specific context, then renders a move-specific prompt.
- **Source of truth:** the companion design doc (`sensorflow_operator_design.md`). This recipe
  must not contradict it. Where the design doc left something open, this recipe pins it or flags
  it — it does not silently decide.
- **Build philosophy:** cheap-first, dumb-on-purpose, minimal edits. v1 only. Every "expensive"
  idea (model-driven selection, summarization agent, smart parent-picking) stays behind a seam and
  is OUT of scope unless explicitly listed as in-scope.
- **Hard carry-over constraint from canonical SensorFlow:** metric keys used anywhere
  (`target_metric`, the cycle, prompts) must match the evaluator's actual metric-dict keys exactly,
  or they are silently ignored. **Live keys (confirmed):** `accuracy` (max), `f1_score_class1`
  (max), `depth` (min), `num_leaf_nodes` (min) — the `max_depth`/`max_leaf_nodes` →
  `depth`/`num_leaf_nodes` rename is **already done**, so there is no rename to avoid.

---

## 2. Component breakdown (units of work)

> The distinct units the design implies, each with a one-line responsibility. Resolutions decided
> with the owner are inlined; see §3 for the flag log and §6 for full signatures.
>
> **NOTE:** §6 had a later simplification pass that **inlined or dropped** several units below
> (`get_frontier`, `build_operation_block`, `load_task_config`, `format_frontier_rows`,
> `frontier_has_enough_to_combine`, `format_selection_reason`, `get_baseline`). Left here for
> traceability; **§6 is authoritative** on what actually gets written.

### Core loop
- **`run_loop`** — drive `num_tries` iterations, calling `run_one_try` each; stop at exactly `num_tries`.
- **`run_one_try`** — orchestrate one try end-to-end (split from `run_loop`; keeps the loop trivial).

### Part 1 — operation selection
- **`choose_operation`** — apply the 1C heuristic, return the op name. Swappable seam (code now, model later).
- **`should_repair`** — probabilistic repair gate (RESOLVED): if a failure exists within the last
  **5 tries**, fire with **p = 0.3** (seeded RNG); else fall through. Stateless; self-limiting
  (can't latch/chain); ~23% of tries at the ~25% fail rate. Raise toward 0.4 if failures pile up.
- **`frontier_has_enough_to_combine`** — boolean (RESOLVED): **≥ 3 distinct frontier points.**
  Smallest size where combine genuinely selects complementary candidates. Controls *eligibility*
  only, not frequency (rate cap parked).

### Part 2 — parent + feedback selection
- **`select_parents`** — the single selector seam: given a *need* (count + eligibility + pool),
  return chosen candidate id(s). Houses the dumb v1 rules.
- **`build_pool`** — the pool an op draws from: frontier for improve/combine; failures within the
  last 5 tries for repair (capped — the cap, not the trigger, is the staleness guard).
- **`pick_frontier_extremes`** — combine's two parents (RESOLVED): draw **two random distinct
  policy metrics**, take frontier-best on each; each parent's `strong_metric` = the metric it was
  picked for. Tie-break (same candidate best on both) → re-draw; pin exact rule when coding.
- **`choose_target_metric`** — improve's target (RESOLVED): **cycle** the metric-policy keys, one
  per improve run, each with its policy direction; advance a single counter. No comparison/
  normalization. Same signature whatever the body (swap to weakest/LLM-picks later moves no call sites).
- **`frontier_neighbors`** — the frontier rows shown to improve. **SCOUT then CHECK IN:** count
  (2–3 vs all vs canonical §6 cap of 5) deferred; report frontier size + existing cap + per-row
  token cost, then owner picks. Do not choose the cap unilaterally.

### Part 3 — prompt construction
- **`render_prompt`** — assemble the shared skeleton (injected task/eval/output config + op block).
- **`build_operation_block`** — dispatch to the correct per-op builder (three; one per locked body).
- **`build_repair_block` / `build_improve_block` / `build_combine_block`** — render each move's body.
- **`build_feedback_context`** — gather the per-move context the builders consume (resolves ids via
  `get_candidate`; attaches scores / failure reason / strong metrics).
- **`format_scores`** — metrics dict → compact score string. **TODO:** no clean reusable helper
  exists (`print_metrics` dumps nested dicts; inline formatters at `runner.py:98`/`:322`) → extract
  a small one lifting the `:322` style: policy keys only, policy order, `name=value`, 4-dp.
- **`format_frontier_rows`** — render improve's "other candidates on the frontier" lines (via `format_scores`).
- **`load_task_config`** — read `task_description` / `eval_description` / `output_format` (I/O).

### Logging
- **`log_try`** — record per try: op fired, the pool, chosen id(s), selection reason.
- **`format_selection_reason`** — short "why this parent" string (now `random` / `cycle:f1` /
  `best-on:num_leaf_nodes`; later the model's stated reason).

### Existing canonical pieces consumed (confirm interfaces, do NOT rebuild)
- **`pareto_frontier_try_ids(rows, metric_policy) -> list[int]`** (`get_pareto.py:164`) — returns
  try_ids, not rows.
- **`mutate(parent, prompt, output_dir, feedback)`** (`py_mutation_engine.py:146`) — already takes
  a prompt string (assembled at `:38`); feed our op-specific prompt in. Minimal adaptation.
- **`evaluate`** → `{status, metrics, failure_reason}`; eval wrapper persists `evaluation_result.json` (`evaluator.py:58/:89`).
- **`record` / `_append_try_row`** (`runner.py:458`/`:375`) — append archive entry + CSV row.
- **`_parse_metrics_from_log`** — parse windowed-eval metrics from the eval log.
- **baseline + loop in `main`** (`runner.py:656`; baseline eval `:428`, record `:458`/`:724`) —
  `setup_run` / `evaluate_baseline` / `finalize_run` slot **into** `main`, not replace it.
- **`_rng = random.Random(42)`** (`runner.py:57`) — single shared instance; thread it, don't make new.
- **`try_dir`** — per-try output directory.

### Non-function artifacts (data/state — surfaced so nothing hides)
- `task_config` — injected strings (`task_description`, `eval_description`, `output_format`).
- selection constants — `REPAIR_HORIZON = 5`, `REPAIR_PROB = 0.3`, `COMBINE_MIN_FRONTIER = 3`.
- `METRIC_POLICY` **(exists, `runner.py:45`)** — plain dict; source of truth for cycle order + directions. Read directly (no accessor functions).
- improve-run counter — integer state `choose_target_metric` advances (on run state, not a global).

---

## 3. Cross-cutting flag log (resolved)

1. ~~Target-metric selection had no rule.~~ **RESOLVED: cycle** the policy keys. (`choose_target_metric`)
2. ~~"Frontier extremes" undefined for >2 metrics.~~ **RESOLVED: two random distinct metrics, best-on-each;** `strong_metric` = metric each was picked for. (`pick_frontier_extremes`)
3. ~~Two windows named "recent failures."~~ **RESOLVED: intentionally two jobs** — trigger = failure within last 5 tries (freshness); pool = failures within that same 5-try horizon (availability, capped).
4. ~~`fmt_scores` key set/format.~~ **RESOLVED: extract small formatter** (no clean reuse), policy keys, policy order, `name=value`, 4-dp.
5. `frontier_neighbors` count — **DEFERRED to scout** (frontier size + existing cap + token cost → owner picks).
6. ~~Selection constants.~~ **RESOLVED:** repair horizon 5, repair p 0.3, combine eligibility ≥ 3.
7. ~~Seam-2 scope.~~ **RESOLVED: scores in improve/combine only; repair score-free** (failed candidate has `metrics=None`, nothing to render).

---

## 4. Data model

**Decision: a small `Candidate` object** — and it is the currency passed between functions (NOT
try_ids). Selection/picker functions return `Candidate`s directly; nothing round-trips through ids.

- Fields: `try_id`, `code: str` (read from disk), `metrics: dict | None`, `status: str`,
  `failure_reason: str | None`. (`strong_metric` field **removed** — combine gets strong metrics as
  explicit args from the picker, the field was never read. `try_id` stays only for logging/lineage,
  not as something other functions match on.)
- `get_candidate` does the one messy unpack (entry lookup, read code file at `entry["candidate"]`,
  pull `entry["result"]["metrics"]` / `failure_reason`); everything downstream uses flat attribute
  access. It's the single place that touches archive-dict internals.
- Raw archive dicts only appear in `should_repair` / `recent_failures` (they scan `status`); a
  `Candidate` is built once an entry is selected.
- Missing entry / unreadable file: **let it raise** — a dict/list/file access raises naturally; do
  **not** write an explicit existence check just to re-raise. (Standing rule below.)

---

## 5. Signature conventions

- `_rng` and `METRIC_POLICY` are module globals (canonical) — not threaded as params.
- `archive` passed explicitly to functions that read it.
- Functions pass **`Candidate`s, not ids.** `try_id` is a field you read, never a key you match on —
  which also makes the int-vs-str question moot (just read whatever it is).
- **Standing rule — robustness is a v1 non-goal.** Anything that can't happen in a correct run
  (missing entry/file, bad op, unsatisfiable pool after its gate) just **raises via natural Python
  access** — no pre-checks, no validators, no try/except, no defensive wrappers. Only conditions that
  legitimately occur in a correct run get real handling (e.g. a candidate failing evaluation).
- **Lazy-pass principle (applied):** no wrapper functions for one-liners, no manifests/registries
  (use `METRIC_POLICY.keys()` directly), no objects to hold a single value, no id→object resolution
  layers. Inline anything used once.

---

## 6. Signatures + contracts

### Data access

```
get_candidate(try_id, archive: list[dict]) -> Candidate
```
- Find the entry by linear scan: the one where `entry["try_id"] == try_id`. From it:
  read the code file at the path in `entry["candidate"]` → `.code`; `entry["result"]["metrics"]`
  → `.metrics` (or `None`); `entry["status"]` → `.status`; `entry["result"]["failure_reason"]`
  → `.failure_reason`; carry `entry["try_id"]` → `.try_id`.
- Reads disk. No matching entry / unreadable file → raises naturally (no explicit check).
  `status != "ok"` resolves fine (`.metrics is None`) — the repair case.
- (`entry["candidate"]` is the candidate path per the archive shape confirmed in Chunk 0 — if it's a
  directory rather than a file path, read the `.py` inside it; confirm in Chunk 0.)
- `get_baseline` — **dropped.** Improve no longer shows a separate baseline line (frontier neighbors
  already give the landscape); one fewer disk read + less prompt clutter.

### Part 1 — operation selection

```
recent_failures(archive) -> list[dict]
```
- Take the **last 5 entries** of `archive` (it's appended in try order — no try_id sorting needed,
  works whether `try_id` is int or string), keep those with `status != "ok"`. None → `[]`.

```
should_repair(archive) -> bool
```
- `recent_failures` non-empty **and** `_rng` draw < `REPAIR_PROB` (0.3). Check non-empty *before*
  drawing (don't perturb the RNG stream on failure-free tries).

```
choose_operation(archive, frontier: list[Candidate]) -> str
```
- `if should_repair(archive): "repair"` → `elif len(frontier) >= COMBINE_MIN_FRONTIER (3): "combine"`
  → `else: "improve"`. Combine length-check inlined. Run start (frontier = just baseline) → `"improve"`.

### Part 2 — parent selection

```
select_parents(op: str, archive, frontier: list[Candidate]) -> list[Candidate]
```
- Handles **repair and improve only** (combine is special — it needs metric labels, so `run_one_try`
  calls `pick_frontier_extremes` directly; routing combine through here too would call the picker
  twice and pick different candidates each time). Returns a 1-element list:
  - `"repair"` → `e = _rng.choice(recent_failures(archive))`; return `[get_candidate(e["try_id"], archive)]`.
  - `"improve"` → `[_rng.choice(frontier)]` (frontier is already `Candidate`s).

```
pick_frontier_extremes(frontier: list[Candidate]) -> tuple[(Candidate, str), (Candidate, str)]
```
- Called **once per combine try, by `run_one_try`** (not via `select_parents`). Draw two distinct
  `METRIC_POLICY` metrics (`_rng`); `a = best_on_metric(m1, frontier)`, `b = best_on_metric(m2, frontier)`,
  return `((a, m1), (b, m2))`.
- **Collision rule (bounded):** if `a` and `b` are the same candidate, fall back to
  `_rng.sample(frontier, 2)`, labelling each with one of the two drawn metrics. Degenerate-frontier
  case; cannot loop.

```
best_on_metric(metric: str, frontier: list[Candidate]) -> Candidate
```
- `max(frontier, key=lambda c: c.metrics[metric])` for a `max` metric, `min(...)` for a `min` metric
  (direction from `METRIC_POLICY`). Natural first-seen tie-break is fine — no explicit tie handling.

*(`get_frontier` dropped — call `pareto_frontier_try_ids` directly where the frontier is built, then
map to `Candidate`s once. `pick_random_*` / `build_pool` are inline in `select_parents`.)*

### Improve target metric

```
choose_target_metric(improve_count: int) -> tuple[str, str]
```
- Cycle `METRIC_POLICY` keys: `keys = list(METRIC_POLICY)`, `key = keys[improve_count % len(keys)]`,
  direction = `METRIC_POLICY[key]` (`"max"`/`"min"`). Return `(key, direction)`.
- **Cycles ALL policy keys** — `accuracy`, `f1_score_class1`, `depth`, `num_leaf_nodes` (confirmed:
  no steer-list exclusion; `accuracy` IS targeted, not only protected).
- `improve_count` is a **plain int** passed in — no `run_state` object. Derive it as the count of
  prior improve tries (cheapest: from the archive/log; or thread a counter from `run_loop`). Resets
  naturally per run.

### Part 3 — prompt construction

```
format_scores(metrics: dict) -> str
```
- **Lift the `runner.py:322` formatting verbatim** — this is the spec; do not reinvent. (For
  reference, expect: `METRIC_POLICY` keys only, policy order, `name=value`, 4-dp for rates / ints
  as-is. If the lifted snippet already does this, use it as-is; the description is what to expect,
  not a second transform to apply on top.)
- `metrics is None` → raises (only called on improve/combine candidates, never repair).

The three block builders render the **locked bodies below verbatim**, slotting in the values. Do not
reword the prompt text. (`{...}` = substitute; everything else is literal.)

```
build_repair_block(fail: Candidate) -> str
```
Renders exactly:
```
TASK: The candidate below failed when it was run. Debug it and return
a working version.

Failed candidate (try {fail.try_id}):
{fail.code}

What went wrong:
{fail.failure_reason}
```

```
build_improve_block(cand: Candidate, target_metric: str, direction: str, neighbors: list[Candidate]) -> str
```
Renders exactly (the neighbor block is one line per neighbor: `try {n.try_id}: {format_scores(n.metrics)}`,
inlined — no separate helper; NO baseline line):
```
TASK: Improve the candidate below on one target metric. Push that
metric as far as you can. Try to hold the other scores where they are,
but you may sacrifice them if it meaningfully helps the target.

Target this run: {target_metric} → {direction}

Current candidate (try {cand.try_id}):
{cand.code}
Current scores: {format_scores(cand.metrics)}

Other candidates on the current frontier:
  try {n.try_id}: {format_scores(n.metrics)}      ← one such line per neighbor

GOALS, in order:
1. Move {target_metric} in the {direction} direction as much as you can.
2. Prefer to keep the other scores steady — but trading some away for
   a real gain on the target is allowed.
3. Make one clear, deliberate change, not a scattershot rewrite.
```

```
build_combine_block(a: Candidate, strong_a: str, b: Candidate, strong_b: str) -> str
```
Renders exactly:
```
TASK: Two candidates below each do something well. Think about whether
their strengths can be combined, and if so, build ONE new candidate
that captures both. This is recombination of ideas — don't blindly
average or paste the two together.

Candidate A (try {a.try_id}) — strong on {strong_a}:
{a.code}
A scores: {format_scores(a.metrics)}

Candidate B (try {b.try_id}) — strong on {strong_b}:
{b.code}
B scores: {format_scores(b.metrics)}

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

No `build_operation_block` dispatcher — `run_one_try`'s op-branch calls the right one directly.

```
render_prompt(operation_block: str) -> str
```
Wraps the block in this exact skeleton (substituting the three module-level config strings and the
prebuilt block):
```
SYSTEM:
  You are improving a candidate program through iterative search.
  Each iteration mutates the program; it is then evaluated and kept
  if it advances the tradeoff frontier.

  THE TASK:
  {task_description}

  HOW YOU ARE JUDGED:
  {eval_description}

  OUTPUT FORMAT:
  {output_format}

  {operation_block}
```
Config (`task_description` / `eval_description` / `output_format`) are imported module-level strings
(Chunk 1), not loaded via a function.

### Core loop + logging

**Minimal splice into existing `main`** — do not restructure it. `setup_run` / `evaluate_baseline`
/ `finalize_run` are **labels for code already in `main`**, not functions to write. Replace only the
loop *body* with `run_one_try`.

```
run_one_try(try_id, archive, improve_count) -> dict
```
- Spine, in order:
  1. `frontier = [get_candidate(tid, archive) for tid in pareto_frontier_try_ids(rows, METRIC_POLICY)]`
     (build rows the way `main` already does).
  2. `op = choose_operation(archive, frontier)`.
  3. Get the candidate(s) + build the block, per op:
     - repair → `chosen = select_parents("repair", archive, frontier)`; `block = build_repair_block(chosen[0])`.
     - improve → `chosen = select_parents("improve", archive, frontier)`;
       `block = build_improve_block(chosen[0], *choose_target_metric(improve_count), neighbors=frontier)`.
     - combine → `(a, sa), (b, sb) = pick_frontier_extremes(frontier)` (called **here**, once);
       `block = build_combine_block(a, sa, b, sb)`; `mutate_parent = a`.
  4. `prompt = render_prompt(block)`.
  5. Set `parent` = the single mutate base: `chosen[0]` for repair/improve, `a` for combine.
  6. `mutate(parent=parent, prompt=prompt, output_dir=…, feedback=…)` — one parent regardless of op
     (combine's real intent lives in the prompt; `mutate` takes one parent per its canonical
     signature). **`--no-mutate` is NOT handled here** — it lives entirely inside canonical `mutate`,
     which returns the parent's source copied verbatim instead of calling the LLM. Operator code is
     unaware of it.
  7. `evaluate(...)` → `record(...)` → `log_try(...)`. Return the recorded entry.
- A candidate eval failure (`status="error"`) is recorded normally (feeds repair), not an error.
- `mutate` wants a parent path (canonical `mutate(parent, …)`); pass the chosen candidate's path.
  Confirm the exact arg form (path vs object) in Chunk 0/8.

```
log_try(try_id, op: str, chosen: list[Candidate]) -> None
```
- Extend the existing `tries.csv` row (writer at `runner.py:375`) with **two columns**: `operation`
  (the op string) and `parent_try_ids` (the chosen candidates' `try_id`s, comma-joined; one id for
  repair/improve, two for combine). Match the existing CSV-writing style; do not add a separate sink.
  **No `format_selection_reason`** — the `"cycle:f1"`-style reason string is premature (nothing reads
  it); add it when something does.

## 7. Dependency graph + build order

> Derived from the §6 signatures. `→` = "calls / depends on". `(exists)` = canonical, already built.
> Build and test **bottom-up**: leaves first (testable in isolation), integrators last. This order is
> what the chunking follows.

### Dependency edges

```
get_candidate            → archive, Candidate, disk          [LEAF]
recent_failures          → archive                           [LEAF]
best_on_metric           → METRIC_POLICY, Candidate           [LEAF]
choose_target_metric     → METRIC_POLICY                      [LEAF]
format_scores            → METRIC_POLICY  (lift runner.py:322)[LEAF]
render_prompt            → module config strings              [LEAF]
log_try                  → existing tries.csv writer          [LEAF-ish, touches canonical]

should_repair            → recent_failures, _rng
pick_frontier_extremes   → best_on_metric, _rng, METRIC_POLICY
choose_operation         → should_repair  (+ inline len(frontier))
select_parents           → recent_failures, get_candidate, pick_frontier_extremes, _rng
build_repair_block       → Candidate                          (no format_scores — score-free)
build_improve_block      → format_scores, Candidate
build_combine_block      → format_scores, Candidate

run_one_try              → pareto_frontier_try_ids(exists), get_candidate, choose_operation,
                           select_parents, choose_target_metric, build_repair/improve/combine_block,
                           render_prompt, mutate(exists), evaluate(exists), record(exists), log_try
```

### Build/test order (tiers — everything in a tier is independent of its siblings)

**Tier 0 — the type.** `Candidate` dataclass. Nothing works without it; trivial.

**Tier 1 — leaves (no internal deps; unit-test each alone).**
`get_candidate`, `recent_failures`, `best_on_metric`, `choose_target_metric`, `format_scores`,
`render_prompt`, `log_try`.

**Tier 2 — depend only on Tier ≤1.**
`should_repair` (→recent_failures) · `pick_frontier_extremes` (→best_on_metric) ·
`build_repair_block` (→Candidate) · `build_improve_block` (→format_scores) ·
`build_combine_block` (→format_scores).

**Tier 3 — depend on Tier ≤2.**
`choose_operation` (→should_repair) · `select_parents` (→recent_failures, get_candidate,
pick_frontier_extremes).

**Tier 4 — the integrator.**
`run_one_try` (→ everything above + canonical `pareto_frontier_try_ids`/`mutate`/`evaluate`/`record`).

**Tier 5 — the splice.**
Wire `run_one_try` into the existing `main` loop body (the only edit to canonical control flow;
`setup_run`/`evaluate_baseline`/`finalize_run` stay as-is in `main`).

### Notes for chunking
- Tiers 0–1 are pure functions over passed-in data (except `get_candidate`'s disk read and
  `log_try`'s CSV append) → fully unit-testable before any integration.
- `format_scores` and `best_on_metric` are the two most-reused leaves (build/verify early).
- `run_one_try` is the only function that can't be meaningfully tested until its tier is reached —
  it *is* the integration test.
- Open items don't block tiering: `frontier_neighbors` count feeds `build_improve_block` (Tier 2) —
  the function builds either way, the cap is a constant filled at scout time; the improve-counter
  source feeds `choose_target_metric` (Tier 1) as an already-resolved `int` arg.

---

## 8. Status

Step 3 complete + a lazy-senior simplification pass applied. Every function has a signature +
contract, or is explicitly folded/dropped. Cut this pass: `get_frontier`, `build_operation_block`,
`load_task_config`, `format_frontier_rows`, `frontier_has_enough_to_combine`,
`format_selection_reason`, `get_baseline` (7 wrappers/one-liners); `Candidate.strong_metric` field;
the `run_state` object (→ plain int); and id→object round-tripping (functions now pass `Candidate`s).
Earlier cuts: `Need`, trivial pickers, `build_feedback_context`. Config is module-level imported
strings. Robustness = let natural Python access raise; no validators.

**§6 is authoritative** where it differs from the §2 breakdown (§2 still lists some now-inlined
names for context).

Still open (intentionally, not blocking): `frontier_neighbors` count (scout → owner picks); improve
cycle-counter source (count prior improve tries from archive, or thread an int). `try_id` int-vs-str
is now moot — it's only ever read, never matched.

---

## 9. Build chunks (Haiku-sized)

> Functions grouped along the §8 dependency order — each chunk the smallest coherent set that
> builds and verifies on its own, leaves first. One chunk = one focused session. New files live in a
> dedicated `operators` module (see Chunk 1); canonical files are touched only where stated.

### Guardrails (apply to EVERY chunk — state once, obey always)
- **DO NOT TOUCH canonical behaviour.** `evaluate`, `mutate`, `pareto_frontier_try_ids`, `record`,
  the archive entry shape, the visibility suite (`summary.json` / `tries.csv` / `pareto.png`), and
  the baseline/loop scaffolding in `main` stay as-is. The ONLY canonical edit in the whole project is
  Chunk 8 (splice one call into `main`'s loop body) + Chunk 1's tiny `tries.csv` column add.
- **DO NOT RENAME anything.** Especially not metric keys — they are `accuracy`, `f1_score_class1`,
  `depth`, `num_leaf_nodes` (already renamed; final). Match existing names; never "tidy" them.
- **NO SCOPE CREEP.** Build only the functions named in the current chunk, with exactly the §6
  signatures. No extra helpers, no validators, no try/except, no abstractions, no "while I'm here."
  Robustness is a v1 non-goal — let natural Python access raise.
- **NO NEW STATE/IDS/WRAPPERS.** No `run_state` object, no id→object resolution layers, no config
  loader, no manifests. Pass `Candidate`s and plain values.
- **If a contract is ambiguous or a needed fact isn't in this doc, STOP and ask** — do not guess.
  Two facts are deliberately deferred (see Chunk notes): `frontier_neighbors` cap, improve-counter
  source. Use the stated placeholder and flag, don't invent.
- **Reuse, don't reinvent:** `format_scores` lifts `runner.py:322` verbatim; `_rng`
  (`runner.py:57`) and `METRIC_POLICY` (`runner.py:45`) are imported, not recreated.

---

> **How to run a chunk (read this first).** Each chunk is **two stages**: **SCOUT** then **IMPLEMENT**.
> The Haiku executor does NOT make design decisions and does NOT guess. In SCOUT it only *reads and
> reports* — it opens the named files/lines, copies out the exact facts asked for, and writes a short
> report. Then it STOPS if the report flags any mismatch. In IMPLEMENT it writes only the named
> functions, exactly to the §6 contract, using the facts from its own scout report. If a fact the
> scout was supposed to find isn't there, STOP and ask — never invent it.

---

### Chunk 0 — Orientation (whole-project scout; no code)
**Model:** Haiku.  **Depends on:** nothing.

**SCOUT (read-only). Open these and copy out the exact text/values:**
- `runner.py:45` → the full `METRIC_POLICY` dict (keys + `"max"`/`"min"` values).
- `runner.py:57` → the `_rng` definition line (confirm it's `random.Random(42)`).
- `runner.py:322` → the metric-formatting code (copy the whole snippet — Chunk 3 lifts it).
- `runner.py:375` → the `tries.csv` row-writing code (copy column list + how a row is written).
- `runner.py:458` → the archive-entry construction (copy the exact dict keys: confirm `try_id`,
  `candidate`, `result`, `parent_try_id`, `status`, …) and note whether `try_id` is an int or a string.
- `runner.py:428` and `:656` → how the baseline is evaluated and where the per-try loop body is.
- `get_pareto.py:164` → confirm `pareto_frontier_try_ids(rows, metric_policy)` signature + that it
  returns a list of try_ids.
- `py_mutation_engine.py:146` → copy the exact `mutate(...)` signature (param names + order).
- `evaluator.py:58` → confirm `evaluate(...)` returns a dict with `status`, `metrics`, `failure_reason`.
- Confirm what `entry["candidate"]` is: a `.py` **file path** or a **directory** (if dir, note the
  `.py` filename inside).

**REPORT (≤15 lines):** the `METRIC_POLICY` dict, `try_id` type, archive-entry keys, the
`runner.py:322` snippet, the `mutate` signature, and the `entry["candidate"]` form.
**STOP CONDITION:** if any value contradicts §1/§4/§6 of this recipe (e.g. metric keys differ,
`mutate` signature differs), STOP and report the contradiction — do not proceed to any chunk.
**No files changed.**

---

### Chunk 1 — Module scaffold + `Candidate` type
**Model:** Haiku.  **Depends on:** Chunk 0.

**SCOUT:** from the Chunk 0 report, confirm: the import path for `_rng` and `METRIC_POLICY` (module
`runner`), and the archive-entry key names. Report the one-line import statement you will use.
**IMPLEMENT** (`operators.py`, new file):
- `from dataclasses import dataclass`; `from runner import _rng, METRIC_POLICY` (use whatever the
  scout confirmed as the real module name).
- `@dataclass class Candidate:` fields exactly — `try_id`, `code: str`, `metrics: dict | None`,
  `status: str`, `failure_reason: str | None`. No methods, no defaults beyond what's needed, no extra fields.
- Constants exactly: `REPAIR_HORIZON = 5`, `REPAIR_PROB = 0.3`, `COMBINE_MIN_FRONTIER = 3`.
- Three module-level config strings `task_description`, `eval_description`, `output_format`, set to
  the placeholder examples + the "be concrete" comment block from the design doc §5. (These are
  EDITED LATER by the user, not by you.)
**VERIFY:** `python -c "import operators"` succeeds; `Candidate(try_id=1, code='x', metrics=None,
status='ok', failure_reason=None)` constructs.

---

### Chunk 2 — Archive leaves: `get_candidate`, `recent_failures`
**Model:** Haiku.  **Depends on:** Chunk 1.

**SCOUT:** from the Chunk 0 report, re-state: the exact archive-entry keys, what `entry["candidate"]`
is (file vs dir), and `try_id` type. Report the one-line "how I read code from an entry" you'll use.
**IMPLEMENT** (`operators.py`) — exactly to §6:
- `get_candidate(try_id, archive)`: linear scan for `entry["try_id"] == try_id`; read code from
  `entry["candidate"]` (per scout: file → read it; dir → read the `.py` inside); map
  `entry["result"]["metrics"]`→`metrics`, `entry["status"]`→`status`,
  `entry["result"]["failure_reason"]`→`failure_reason`, `entry["try_id"]`→`try_id`. No try/except.
- `recent_failures(archive)`: `[e for e in archive[-REPAIR_HORIZON:] if e["status"] != "ok"]`.
**VERIFY (unit test, fake archive of dicts + temp code files):** `get_candidate` returns a populated
`Candidate`; an entry with `status="error"` yields `.metrics is None`; `recent_failures` returns only
non-ok among the last 5, `[]` when none.

---

### Chunk 3 — Metric leaves: `format_scores`, `best_on_metric`, `choose_target_metric`
**Model:** Haiku.  **Depends on:** Chunk 1.

**SCOUT:** paste the `runner.py:322` snippet from the Chunk 0 report; state in one line how it formats
(so IMPLEMENT lifts it, not reinvents). State `list(METRIC_POLICY)` order from the report.
**IMPLEMENT** (`operators.py`) — exactly to §6:
- `format_scores(metrics)`: lift the `runner.py:322` logic verbatim. Do NOT add your own formatting.
- `best_on_metric(metric, frontier)`: `max(frontier, key=lambda c: c.metrics[metric])` if
  `METRIC_POLICY[metric] == "max"`, else `min(...)`. No tie handling.
- `choose_target_metric(improve_count)`: `keys = list(METRIC_POLICY); k = keys[improve_count %
  len(keys)]; return (k, METRIC_POLICY[k])`.
**VERIFY:** `format_scores` output equals the existing renderer on one sample dict; `best_on_metric`
returns the true extreme for one `max` and one `min` metric; `choose_target_metric(0,1,2,3,4)` cycles
keys in policy order and wraps.

---

### Chunk 4 — Selection logic: `should_repair`, `pick_frontier_extremes`, `choose_operation`
**Model:** Haiku.  **Depends on:** Chunks 2, 3.

**SCOUT:** confirm from the report: `_rng` is the shared seeded instance (use `_rng.random()`,
`_rng.sample`, `_rng.choice` — never bare `random.*`). No new scouting otherwise; report "ready".
**IMPLEMENT** (`operators.py`) — exactly to §6:
- `should_repair(archive)`: `fails = recent_failures(archive); return bool(fails) and _rng.random()
  < REPAIR_PROB`. (The `bool(fails)` short-circuits BEFORE the draw — required.)
- `pick_frontier_extremes(frontier)`: `m1, m2 = _rng.sample(list(METRIC_POLICY), 2);
  a = best_on_metric(m1, frontier); b = best_on_metric(m2, frontier)`; if `a is b` →
  `a, b = _rng.sample(frontier, 2)`; `return ((a, m1), (b, m2))`.
- `choose_operation(archive, frontier)`: `if should_repair(archive): return "repair";
  elif len(frontier) >= COMBINE_MIN_FRONTIER: return "combine"; else: return "improve"`.
**VERIFY (seed `_rng` for determinism):** `should_repair` is `False` with no recent failures and never
calls `_rng` then; with a recent failure it's `True` ~30% over many draws; `pick_frontier_extremes`
returns two distinct candidates + two metric names; `choose_operation` hits each branch on crafted inputs.

---

### Chunk 5 — `select_parents` (repair + improve only)
**Model:** Haiku.  **Depends on:** Chunks 2, 4.

**SCOUT:** none new — report "ready; combine is NOT handled here (run_one_try calls
pick_frontier_extremes directly)".
**IMPLEMENT** (`operators.py`) — exactly to §6:
- `select_parents(op, archive, frontier)`: if `op == "repair"`: `e = _rng.choice(recent_failures(
  archive)); return [get_candidate(e["try_id"], archive)]`. If `op == "improve"`:
  `return [_rng.choice(frontier)]`. (No `combine` branch.)
**VERIFY:** repair returns a 1-list holding a failed candidate; improve returns a 1-list holding a
frontier candidate.

---

### Chunk 6 — Prompt blocks + `render_prompt`
**Model:** Sonnet (prompt text must match the locked bodies exactly).  **Depends on:** Chunks 1, 3.

**SCOUT:** copy the four locked text blocks from §6 of THIS recipe (repair body, improve body, combine
body, skeleton) into the report, so IMPLEMENT renders them verbatim. State the neighbor-row format:
`  try {n.try_id}: {format_scores(n.metrics)}`, one line per neighbor.
**IMPLEMENT** (`operators.py`) — render the §6 bodies verbatim, substituting only `{...}` fields:
- `build_repair_block(fail)` → repair body.
- `build_improve_block(cand, target_metric, direction, neighbors)` → improve body; build the neighbor
  block by joining one row per neighbor; **no baseline line**.
- `build_combine_block(a, strong_a, b, strong_b)` → combine body.
- `render_prompt(operation_block)` → the skeleton, substituting the three module-level config strings.
- **DEFERRED FACT — `frontier_neighbors` cap:** the *caller* (`run_one_try`) passes the neighbor list;
  these builders render whatever they're given. Capping happens at the call site, defaulting to 5
  (named constant `NEIGHBOR_CAP = 5`); flag this for owner confirmation, do not decide it here.
**VERIFY:** each builder's output equals the locked body with values slotted (diff against the §6
text); repair shows no scores; improve shows N neighbor rows and no baseline; `render_prompt` wraps
the block with task/eval/output strings.

---

### Chunk 7 — Logging: `log_try`
**Model:** Haiku.  **Depends on:** Chunk 1.

**SCOUT:** paste the `runner.py:375` row-writing code from the Chunk 0 report; state the current
column list and exactly how a row is appended (so IMPLEMENT matches the style and adds two columns).
**IMPLEMENT:**
- `log_try(try_id, op, chosen)` in `operators.py`: compute `parent_try_ids = ",".join(str(c.try_id)
  for c in chosen)`; write `op` and `parent_try_ids` into the per-try row.
- **Minimal `runner.py:375` edit:** add two columns — `operation`, `parent_try_ids` — to the existing
  writer. Touch nothing else in that function.
**VERIFY:** a generated `tries.csv` row gains `operation` + `parent_try_ids`; all existing columns
unchanged and in the same order.

---

### Chunk 8 — Integrator `run_one_try` + splice into `main`
**Model:** Sonnet (only integration point; edits canonical control flow).  **Depends on:** 2–7 +
canonical `pareto_frontier_try_ids` / `mutate` / `evaluate` / `record`.

**SCOUT:** from the Chunk 0 report, copy: the exact `mutate(...)` signature, how `main`'s current loop
body builds `rows` for `pareto_frontier_try_ids`, how `record(...)` is called, and how `main` numbers
try_ids. Report the exact current loop-body code you will replace.
**IMPLEMENT** to the §6 `run_one_try` spine, in order:
1. `frontier = [get_candidate(tid, archive) for tid in pareto_frontier_try_ids(rows, METRIC_POLICY)]`.
2. `op = choose_operation(archive, frontier)`.
3. per op: repair → `chosen = select_parents("repair", …)`, `block = build_repair_block(chosen[0])`;
   improve → `chosen = select_parents("improve", …)`, `block = build_improve_block(chosen[0],
   *choose_target_metric(improve_count), neighbors=frontier[:NEIGHBOR_CAP])`; combine →
   `(a,sa),(b,sb) = pick_frontier_extremes(frontier)`, `block = build_combine_block(a,sa,b,sb)`.
4. `parent = chosen[0]` (repair/improve) or `a` (combine).
5. `prompt = render_prompt(block)`.
6. `mutate(parent=<parent's path per scout signature>, prompt=prompt, …)` — `--no-mutate` is NOT
   handled here (canonical `mutate` owns it).
7. `evaluate` → `record` → `log_try(try_id, op, chosen_or_[a,b])`. Return the recorded entry.
- **Splice:** replace ONLY `main`'s existing per-try loop body with a `run_one_try(...)` call. Leave
  setup, baseline (`try_000`) bootstrap, and finalize/visibility exactly as they are.
- **DEFERRED FACT — `improve_count` source:** derive it as the number of prior entries whose logged
  `operation == "improve"` (read from the archive/log); flag for owner confirmation. Do not invent a
  `run_state` object.
**VERIFY:** (1) `--no-mutate` smoke run completes `num_tries`, records `try_000` + N entries, and
still produces `summary.json`/`tries.csv`/`pareto.png`. (2) A short real run shows all three ops
appearing in the `operation` column and failed candidates recorded with `status="error"` and reused
by repair.

---

### Chunk order recap
`0 orient → 1 scaffold+type → 2 archive leaves → 3 metric leaves → 4 selection → 5 select_parents
→ 6 prompts → 7 logging → 8 integrate+splice`. 2/3 independent; 5 needs 4; 6/7 independent of 4/5;
all converge at 8. Each chunk: SCOUT (report, stop on mismatch) → IMPLEMENT (named functions only) →
VERIFY.
