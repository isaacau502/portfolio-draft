# Scout Instructions — SensorFlow Iterative Loop (READ-ONLY)

You are scouting an existing codebase to report ground-truth facts. **Do NOT modify any files.** Do NOT propose designs. Report only what the code actually says, with file paths and line numbers. If something doesn't exist, say "not found" — do not guess.

Run these as **4 independent chunks**. Each chunk is self-contained — you do not need context from the others. Output each chunk's findings before moving to the next. Keep paste snippets minimal (only the lines asked for).

---

## CHUNK 1 — Archive & path resolution (in `runner.py`)

1. Find where a single try runs today (baseline + N independent edits + eval). Paste ONLY the ~10-15 lines spanning: where `mutate` is called, where the candidate `.py` is written to disk, and where the evaluator is invoked. I want to see which local variables coexist in that scope — specifically whether the candidate `.py` path is held as a variable at the moment the eval result is available.
2. Confirm the literal per-try directory layout. Is the candidate always at `tries/try_NNN/mutated_artifact/<filename>.py` and the result at `tries/try_NNN/mutated_artifact_eval/evaluation_result.json`? Report exact folder names and the exact candidate filename.
3. Report the zero-pad width of the try id (e.g. `try_001` = 3 digits). Find where the try-dir name is constructed.
4. Is `evaluation_result.json` written automatically inside the eval code path, or only by manual invocation? Point to the write site.

**Feasibility check (end of chunk):** Based on what you found, answer in 2-3 sentences: can the runner store the candidate `.py` path in an in-memory archive entry AT THE TIME it appends the eval result, without deriving the path from the result object afterward? If not, what's the blocker?

---

## CHUNK 2 — Evaluator return shape & confusion matrix

Target the evaluator file (e.g. `evaluator.py` / `evaluate_classifier_sensorflow.py`).

5. Paste the evaluator's function signature and its `return` statement(s) — the actual dict it constructs in code (not the JSON file). Show where `metrics`, `status`, and `failure_reason` are assembled.
6. Find where the confusion matrix is computed (the 2x2 array or raw counts, just before the PNG is drawn). Paste that line / those lines so I can see what variable holds the counts.
7. Is there currently ANY try/except around the candidate's train+eval execution, or does a bad candidate crash uncaught? Report what exists.

**Feasibility check (end of chunk):** In 2-3 sentences: can four raw confusion counts (`true_neg, false_pos, false_neg, true_pos`, positive class = car crash) be added to the returned `metrics` dict from the variable found in (6), with no new computation? And can a try/except be wrapped around ONLY the candidate-execution span to return `status="error"` + `failure_reason=str(e)` instead of crashing? Note any obstacle.

---

## CHUNK 3 — Mutation prompt & mutate contract

Target the mutation engine file (e.g. `py_mutation_engine.py` / `mutation_engine.py`).

8. Paste `build_mutation_prompt()`'s signature and current parameter list, plus the section where baseline source is appended between `---` delimiters. I need the exact insertion point.
9. Paste `mutate()`'s current signature.

**Feasibility check (end of chunk):** In 2-3 sentences: can a new `feedback: str` parameter be threaded through `mutate()` into `build_mutation_prompt()` and inserted as a labeled section BETWEEN the rule bullets and the `---`-delimited baseline source, without restructuring the existing prompt? Note any obstacle.

---

## CHUNK 4 — get_pareto input contract

Target `get_pareto.py`.

10. Paste `get_pareto`'s (and/or `pareto_frontier_try_ids`'s) signature and the exact structure it expects for its input rows (the scout notes suggested `list[(try_id, metrics_dict)]` plus a `metric_policy` dict — confirm or correct). Show how it reads each row.

**Feasibility check (end of chunk):** In 2-3 sentences: given an in-memory list of archive entries, can the runner build `get_pareto`'s expected input by filtering to `status=="ok"` and reshaping — WITHOUT modifying `get_pareto` itself? Note the exact shape the runner must produce.

---

## Final output

After the 4 chunks, give a one-paragraph summary: which of the planned additions are clean drop-ins vs. which hit an obstacle that needs a design revisit. List any clarifying questions you'd want answered before implementation.
