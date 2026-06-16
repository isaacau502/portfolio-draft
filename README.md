# SensorFlow Iterative Loop — Build Recipe (paste-by-paste)

Paste **CHUNK 0** first to set context. Then paste **CHUNK 1 → 7** one at a time, verifying each before the next. Each step chunk assumes the agent still has CHUNK 0 in the session.

---

## CHUNK 0 — Context (paste first, don't implement anything)

You are working on **SensorFlow**, a lightweight orchestration layer that improves a machine-learning candidate via LLM-driven mutation. A "candidate" is a Python file that does feature selection + decision-tree training. An evaluator runs the candidate and returns metrics. We are adding an **iterative loop**: instead of mutating a fixed baseline N independent times, each try picks a parent from the current Pareto frontier, mutates it with feedback about what's been tried, evaluates, and records the result. There is no acceptance gate — every result (including failures) is recorded; the Pareto frontier is the implicit quality filter.

**Key facts you must rely on (don't re-derive):**
- Pareto function: `pareto_frontier_try_ids(rows, metric_policy)` in `get_pareto.py`. Input `rows` = `list[(try_id:int, metrics_dict:dict)]`. **Returns a list of try_ids (ints)**, NOT candidates. Do not modify it.
- `metric_policy` is `dict[str, str]`, values `"max"`/`"min"`, keys are exact metric names (e.g. `{"accuracy": "max", "f1_score_class1": "max"}`).
- Candidate file path: `tries/try_{id:03d}/candidate.py`. Eval result: `tries/try_{id:03d}/mutated_artifact_eval/evaluation_result.json` (written automatically by the evaluator on success).
- An eval result dict looks like: `{"status": "ok"|"error", "metrics": {...}|None, "failure_reason": str|None, "candidate_id": ..., "adapter_key": ...}`.

**Global rules for every step:**
- Make additive/minimal changes. Do not refactor data layers or core objects beyond what each step says.
- Before writing code in a step, confirm it's feasible against the real code and report any mismatch with what the step claims. If a path/signature/line differs from what the step states, STOP and ask — do not silently adapt.
- You may decide local details (variable names, number formatting). You may NOT change contracts, control flow, or any "DON'T" listed.
- When something is ambiguous, ask one clarifying question rather than guessing.

Acknowledge you've read this and wait for CHUNK 1. Do not write code yet.

---

## CHUNK 1 — Evaluator: catch candidate-training failures

**Goal:** today a broken candidate crashes the whole evaluator uncaught. Make candidate failures return an error dict so the loop can continue, while real framework bugs still crash loudly.

**Do:**
- In the evaluator (`evaluate_classifier_sensorflow.py`), wrap ONLY the candidate train+eval execution span (around lines 58-67: the `importlib` module load, `module.train_candidate(context)`, and the `clf_path` resolution) in `try/except Exception as e`.
- On exception, print one line `print(f"[eval] candidate failed: {str(e).splitlines()[0]}")` and return:
  ```python
  return {"status": "error", "metrics": None, "failure_reason": str(e),
          "candidate_id": candidate_id, "adapter_key": adapter_key}
  ```
  (Match the shape of the evaluator's existing error returns — they already use `metrics=None`.)

**Don't:**
- Don't wrap data loading, success-path metric computation, or anything outside candidate execution — those must still crash loudly.
- Don't add a `failure_type` field or classify failures. All failures → `status="error"`; detail lives only in `failure_reason`.
- Don't use `metrics={}`. Use `None`.

**Verify:** a deliberately broken candidate now returns the error dict instead of crashing; a good candidate still returns `status="ok"` unchanged.

**Feasibility first:** confirm lines 58-67 are the candidate-execution span and `candidate_id`/`adapter_key` are in scope there; confirm existing error returns use `metrics=None`. Report before coding.

---

## CHUNK 2 — Evaluator: capture confusion-matrix counts

**Goal:** surface the class-imbalance picture (rare class = car crash) to feedback later, by adding four raw confusion-matrix counts into the metrics dict. No new computation — the counts already exist in the eval log.

**Do:**
- Extend `_parse_metrics_from_log()` (evaluator, ~line 117) to also parse the confusion-matrix line from `evaluation_log.txt` (written ~line 322) and add four integer keys to the returned `metrics` dict (present before the success return ~line 127):
  `nb_segments_class0_as_class0`, `nb_segments_class0_as_class1`, `nb_segments_class1_as_class0`, `nb_segments_class1_as_class1`.

**Don't:**
- Don't rename these keys (no `true_pos`/etc.).
- Don't add them to `metric_policy` — they're diagnostic only, never Pareto objectives.
- Don't recompute anything — only parse existing log values.

**Verify:** one successful eval now has the four `nb_segments_*` integer keys in `metrics`; Pareto behavior is unchanged (because they're not in the policy).

**Feasibility first:** report the exact format of the CM line in the log you'll parse, and confirm those values appear on every successful run.

---

## CHUNK 3 — Archive entry shape (runner-side)

**Goal:** introduce the in-memory archive — there is none today. It's a list of dict entries built during the run.

**Do:**
- In the runner (`runner.py`), represent each archive entry as a dict:
  ```python
  {"try_id": int, "candidate": <candidate .py path>, "result": <eval result dict>, "parent_try_id": int | None}
  ```
- `candidate` = the candidate path the runner already holds at the eval call site (around runner.py:262-270, the `candidate_path` local). Don't derive it from the result.
- `parent_try_id` = the selected parent's try_id; `None` for the baseline (try_000, the root).

**Don't:**
- Don't build a tree/graph object — lineage stays implicit via `parent_try_id`.
- Don't store fitted `.pkl` trees in the entry.

**Verify:** you can construct an entry from a real eval result; the baseline entry has `parent_try_id=None`.

**Feasibility first:** confirm `candidate_path` and the eval result dict coexist as locals at the append point (~runner.py 262-270); report the exact variable names there.

---

## CHUNK 4 — `select_parent(successful, metric_policy)`

**Goal:** each try, pick the parent to mutate — uniformly at random from the current Pareto frontier of successful entries.

**Do:**
- Add `select_parent(successful: list[dict], metric_policy: dict) -> dict` (returns the chosen archive ENTRY):
  1. `rows = [(e["try_id"], e["result"]["metrics"]) for e in successful]`
  2. `frontier_ids = pareto_frontier_try_ids(rows, metric_policy)` (returns try_ids)
  3. `chosen_id = random.choice(frontier_ids)` using a module-level RNG seeded with **42**
  4. look `chosen_id` up in the archive, return that entry
- Turn 1 (only baseline in `successful`): frontier = [baseline] → returns baseline naturally; no special-casing.
- Zero successful entries (only if the baseline itself failed eval): **crash loudly** with a clear message like `"baseline failed eval, cannot start"`.

**Don't:**
- Don't modify `pareto_frontier_try_ids`.
- Don't return the raw try_id int — return the entry (caller needs the candidate path).
- Don't add other strategies, flags, or a registry. Exactly one rule.
- Don't guard the empty case into a silent recovery — it must crash with the clear message, not an opaque `random.choice([])` IndexError.

**Note:** seed=42 makes parent selection reproducible only; it does NOT make the whole run deterministic (tree training may vary).

**Verify:** with a hand-built 3-entry `successful` (one dominated), returns only from the 2 frontier members; with one entry returns it; empty raises the clear error.

**Feasibility first:** confirm `pareto_frontier_try_ids(rows, metric_policy)` accepts `rows` as `list[(int, dict)]` and returns `list[int]`, and that `e["result"]["metrics"]` is the right path. Report.

---

## CHUNK 5 — `summarize_feedback(archive, metric_policy, parent_try_id)`

**Goal:** render the feedback string injected into the mutation prompt. Pre-rendered plain text, three blocks, exact format below.

**Do:** implement `summarize_feedback(archive: list[dict], metric_policy: dict, parent_try_id: int) -> str` returning EXACTLY this structure:

```
PRIOR ATTEMPTS / FEEDBACK:

Baseline: accuracy=0.9100 f1_score_class1=0.5500
Best so far: accuracy=0.9391 f1_score_class1=0.5505

Pareto frontier (3 candidates):
  try_004: accuracy=0.9391 f1_score_class1=0.4923  ← mutating this
  try_002: accuracy=0.9210 f1_score_class1=0.5505
  try_000: accuracy=0.9100 f1_score_class1=0.5500   (baseline)

Parent diagnostics (try_004):
  nb_segments_class1_as_class1=12 nb_segments_class1_as_class0=0 nb_segments_class0_as_class1=47 nb_segments_class0_as_class0=703
```

Rules:
- Header `PRIOR ATTEMPTS / FEEDBACK:` + blank line.
- Anchor: `Baseline:` line = try_000's policy metrics; `Best so far:` line = best value PER policy metric across successful entries (each metric's best may come from a different candidate — don't pick one "best candidate").
- Frontier: `Pareto frontier (<count> candidates):`, one indented row per member `try_NNN: <policy metrics only>`. Mark the parent with trailing `  ← mutating this`; mark try_000 with trailing `   (baseline)`.
  - **Display cap = 5.** If the frontier has >5 members, sort by the first key in `metric_policy`, show top 5, append an indented `+K more`. (This cap is DISPLAY ONLY — it does not affect selection.)
- Parent diagnostics: `Parent diagnostics (try_NNN):` then one indented line of the four raw `nb_segments_*` counts as-is. Parent only.
- Numbers: rates/scores 4 decimals; counts raw ints. Single blank line between blocks; order anchor → frontier → diagnostics.

**Don't:**
- Don't emit JSON or use a "Previous feedback:" line.
- Don't put non-policy metrics (depth, leaves, CM counts) in frontier rows — policy metrics only there; CM counts appear only in the diagnostics block.
- Don't render diagnostics for any member other than the parent.

**Verify:** feed a hand-built archive (baseline + 2 tries, known metrics) + a parent_try_id → output matches the template structure exactly.

**Feasibility first:** confirm you can map `pareto_frontier_try_ids` output back to entries for the frontier rows, and confirm "first key in `metric_policy`" is a usable sort key. Report.

---

## CHUNK 6 — Thread feedback into the mutation engine

**Goal:** get the feedback string into the prompt, in the right position, through all the layers.

**Do (in `py_mutation_engine.py`):**
- Add a `feedback: str` parameter to THREE layers, passing it down: `mutate → generate_py_candidate → _build_prompt`.
- In `_build_prompt(baseline_source, mutation_instruction, feedback)`, insert the feedback block AFTER the mutation-objective line (`f"- Mutation objective: {mutation_instruction}\n\n"`) and BEFORE the `"Baseline candidate source:\n"` label. Insert the Chunk-5 string verbatim (already fully rendered).
- **First iteration has no history:** when there's no feedback, insert the literal line `No feedback yet (first iteration).` in that same position (rather than an empty block).
- Rename `mutate`'s first parameter `baseline_candidate → parent`: `mutate(parent: Path, prompt: str, output_dir: Path, feedback: str) -> Path`. Update call sites.

**Don't:**
- Don't restructure the existing prompt — this is one additive inserted section.
- Don't re-render/reformat feedback inside the engine — Chunk 5 already produced the final string.
- Don't stop at two layers — all three get the parameter.

**Verify:** `mutate(parent_path, prompt, out_dir, feedback="...")` puts the block in the right spot; with no feedback the prompt shows `No feedback yet (first iteration).` there.

**Feasibility first:** confirm the three-layer chain and the exact insertion line; report all call sites of `mutate` affected by the rename.

---

## CHUNK 7 — The iterative loop (replaces the old IID flow)

**Goal:** wire Chunks 1-6 into the loop, REPLACING the old "baseline + N independent edits" control flow.

**Do (in `runner.py`):**
- Eval the baseline (try_000); build its archive entry (Chunk 3) with `parent_try_id=None`.
- For `try_id` in `1..num_tries`:
  1. `successful = [e for e in archive if e["status"] == "ok"]`
  2. `parent = select_parent(successful, metric_policy)`
  3. `feedback = summarize_feedback(archive, metric_policy, parent["try_id"])`
  4. `candidate_path = mutate(parent["candidate"], prompt, output_dir=<try dir for try_id>, feedback=feedback)`
  5. eval the candidate (existing wiring) → result dict
  6. append the archive entry (Chunk 3) with `parent_try_id = parent["try_id"]`
- Stop after exactly `num_tries` iterations. No early exit.

**Don't:**
- Don't keep the old IID path alongside this or behind a flag — REPLACE it. Reuse the existing mutate-call/eval-call wiring; delete the independent-from-baseline control flow around them.
- Don't read `evaluation_result.json` back from disk mid-loop — use the in-memory archive. (The per-try JSON still gets written automatically by the evaluator; the loop just doesn't read it back.)
- Don't add acceptance / `improved()` logic — append every result, including failures.

**Verify:** a short run (`num_tries=3`) on the real task → archive has 4 entries (baseline + 3); feedback references real parents; a failed candidate is recorded and the loop continues; frontier evolves.

**Feasibility first:** report where the old IID control flow lives and its boundaries; confirm `num_tries`, `prompt`, `metric_policy`, baseline path are available as run inputs.

---

## OUT OF SCOPE (do not do during this build)
- Renaming `max_depth`/`max_leaf_nodes` to fitted-value names (separate task; touches every `metric_policy`).
- Frontier duplicate handling, failure-type classification, chosen-vs-fitted cap surfacing, per-direction history, a CLI flag for the cap, or multiple selection strategies.



Here's the prompt to hand Sonnet:

---

**Context:** We just implemented the SensorFlow iterative mutation loop (CHUNKS 1-7). Before trusting it, I want smoke tests that verify the loop behaves as intended — both by reading the code and by running it.

**Intended behavior to verify:**

The loop replaces the old independent-N-edits flow with iterative frontier-based mutation. Each run: evaluate baseline (try_000), then for `num_tries` iterations — sample a parent uniformly from the current Pareto frontier of successful entries, build a feedback string about prior attempts, mutate that parent with the feedback, evaluate, and append the result (success *or* failure) to an in-memory archive. No acceptance gate — every result is recorded; the Pareto frontier is the only quality filter. Parent selection is seeded (42). The frontier is computed over all successful entries; failures (`status="error"`) are excluded from selection but still recorded. Baseline is `parent_try_id=None`; each mutation records its actual parent's try_id. Feedback shows baseline + best-so-far anchor, the frontier (parent marked, capped at 5 display rows), and the parent's confusion-matrix diagnostics. The loop stops at exactly `num_tries` — no early exit.

**Two kinds of checks I want:**

1. **Static (read the code):** confirm the control flow matches the above — especially that the old IID path is fully gone (no leftover independent-baseline loop), failures are appended not dropped, `get_pareto` is unmodified, and feedback is inserted in the prompt between the mutation-objective line and the baseline source. Report anything that contradicts the intended behavior.

2. **Runtime (run it):** design and run smoke tests covering at least — a short clean run (e.g. `num_tries=3`) producing baseline + 3 archive entries; a run where a candidate fails eval (inject/simulate one) showing the failure is recorded and the loop continues; parent selection only ever drawing from frontier members; and feedback for try ≥1 referencing a real parent with the correct structure. Use a fast/dummy evaluator or task if a full run is slow.

You have agency on test design and how to simulate failures — propose your approach, then implement and run. Flag any behavior that diverges from the intended description above rather than adjusting the tests to pass.
