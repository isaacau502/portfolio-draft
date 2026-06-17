# Scout: map the run-output directory machinery

You're loading context for an upcoming change to how SensorFlow organizes a run's output directory. **Do not make any change this pass.** Your only job is to read the code and report an accurate map of how `runs/<run_name>/` is currently produced.

## Background
SensorFlow is an LLM-driven mutation loop. The runner iterates: select a parent candidate, mutate it into a new candidate `.py` (via a Bedrock LLM call), evaluate it, append the result to an in-memory archive. Each iteration is a "try." There is also a `--no-mutate` mode used to smoke-test the framework without LLM calls. Every run writes a directory tree under `runs/<run_name>/`.

## What to map and report
For each item give `file:line` and a one-line description.

1. **Run-root files** — what gets written directly under `runs/<run>/` (e.g. a baseline source snapshot, a run summary, a frontier plot), and where each write happens.
2. **Baseline handling** — every place the baseline is treated differently from a normal try: its source snapshot, its result directory, its candidate-output directory.
3. **Per-try structure** — for a normal try, every file and dir created under it: the candidate source, the raw LLM response, the runner's result JSON (and the subdir it sits in), and the candidate's own output dump dir.
4. **Dump-dir name derivation** — the exact expression that names the directory the candidate writes its ~18 output files into. Report whether that name is computed in **one** place (then passed around via `context["output_dir"]`) or **reconstructed independently** anywhere else, e.g. the metrics parser rebuilding the path. *This is the single most important thing to get right.*
5. **Metrics parser** — which file it reads to extract the canonical metrics, and from which directory.
6. **`--no-mutate` path** — the flag plumbing, the branch it takes, and every directory it creates that a normal run does not.
7. **Readers of these paths** — grep the repo for each of: `baseline_eval`, `seed_candidate_eval`, `mutated_artifact_eval`, `candidate_eval`, `mutated_artifact`, `seed_candidate.py`, `_eval`, `evaluation_result.json`, `output_dir`. For every hit say whether it **writes**, **reads**, or just **mentions** the path. Include tests, docs, plotting code, and any summary writer.

## Known anchors (from an earlier session — VERIFY each, they may have moved)
- `runner.py:48` — `try_dir()` builds `tries/try_{id:03d}`
- metrics parser `_parse_metrics_from_log` ≈ `runner.py:136` — reads `outdir/"evaluation_log.txt"`
- evaluator `evaluate_classifier_sensorflow.py:78` ≈ `outdir = candidate_path.parent / (candidate_path.stem + "_eval")`
- mutation engine `py_mutation_engine.py:141` ≈ `mutate(parent, prompt, output_dir) -> Path`

## Output
A concise map covering items 1–7, each finding as `path:line — write/read/mention — note`. End with a short **"Surprises"** list: anything that contradicts the picture above, especially any second place the dump-dir name is derived. **Make no edits.**




# Edit: reorganize the run-output directory layout

The current layout has already been mapped (scout pass; findings are in this thread). Now apply the reorg below. Make the smallest edits that achieve the target — prefer changing path construction in framework code over moving files after the fact.

## Scope
- You **may** change: directory names and placement chosen by the runner and evaluator framework; where the runner writes `evaluation_result.json`; how the candidate output-dir name is derived.
- You **must NOT** change the contents or internal filenames of the candidate's output dump (~18 files: `decision_tree_*`, `evaluation_log.txt`, `log_*`, `confusion_matrix_snippets*`, `*_with_prediction.csv`, etc.). Only the **directory name** changes.
- You **must NOT** edit the logic of LLM-generated candidate scripts. They already honor `context["output_dir"]`; that contract stays intact.

## Target layout

```
runs/<run>/
├── summary.json                 # unchanged, stays at root
├── pareto.png                   # unchanged, stays at root
└── tries/
    ├── try_000/                 # baseline is now just try 0
    │   ├── candidate.py
    │   ├── evaluation_result.json   # runner result, FLAT (no subdir)
    │   └── eval_outputs/            # the ~18-file dump, contents UNCHANGED
    ├── try_001/
    │   ├── candidate.py
    │   ├── llm_response.json        # ABSENT under --no-mutate
    │   ├── evaluation_result.json
    │   └── eval_outputs/
    └── try_NNN/ …
```

## Mapping (old → new)

| old | new |
|---|---|
| root `seed_candidate.py` | `tries/try_000/candidate.py` |
| `baseline_eval/evaluation_result.json` | `tries/try_000/evaluation_result.json` |
| `seed_candidate_eval/` | `tries/try_000/eval_outputs/` |
| `tries/try_NNN/mutated_artifact_eval/evaluation_result.json` | `tries/try_NNN/evaluation_result.json` |
| `tries/try_NNN/candidate_eval/` | `tries/try_NNN/eval_outputs/` |
| `mutated_artifact/`, root `mutated_artifact_eval/` | **deleted** (see edit E) |

`summary.json` and `pareto.png` stay at the run root, unchanged.

## Edits (in order)

**A. Fold baseline into `try_000`.** Route the baseline through the same try machinery as a try with id `0`. Its source becomes `tries/try_000/candidate.py` (drop the root `seed_candidate.py` snapshot). Remove the `baseline_eval/` special case.

**B. Flatten the runner result.** Write `evaluation_result.json` directly to `tries/try_{id:03d}/evaluation_result.json`. Remove the `mutated_artifact_eval/` (and `baseline_eval/`) intermediate result subdir.

**C. Rename the dump dir to `eval_outputs/`.** Change the dump-dir derivation to a fixed name under the candidate's own try dir, e.g. `outdir = candidate_path.parent / "eval_outputs"` instead of `<stem>_eval`. Since each candidate now lives at `tries/try_{id:03d}/candidate.py`, this yields `tries/try_{id:03d}/eval_outputs/` for baseline and tries alike. **Rename no file inside this dir.** If the scout found the name derived in more than one place, change all of them identically.

**D. Delete dead scaffolding.** Remove creation of `mutated_artifact/` and the root `mutated_artifact_eval/` entirely; they no longer exist in either mode.

**E. Rewrite `--no-mutate` to flow through the normal loop.** It must produce `tries/try_001/`, `try_002/`, … with the identical structure as a real run. Differences only:
- At the mutate step, **copy the selected parent's `candidate.py` to `tries/try_{id:03d}/candidate.py`** instead of calling Bedrock. Parent selection is an existing, separate loop step that already runs every iteration — reuse it; add no new selection logic.
- **Skip writing `llm_response.json`.**
- Everything downstream (evaluate → write `evaluation_result.json` → `eval_outputs/` → archive append → Pareto) is shared with the real path, unchanged.
Delete the old single-shot `--no-mutate` branch that used `mutated_artifact*`.

**F. Fix the readers.** Update every read-side reference the scout found (plotting that loads images from the dump dir, any `summary.json` path fields, tests, docs) to the new paths. The metrics parser must still read `evaluation_log.txt`, now from `eval_outputs/`.

## Verify before finishing
1. Run root holds only `summary.json` and `pareto.png`.
2. `tries/try_000/` holds `candidate.py`, `evaluation_result.json`, `eval_outputs/`; no `baseline_eval/`, `seed_candidate_eval/`, or root `seed_candidate.py` remain anywhere.
3. A normal try holds exactly `candidate.py`, `llm_response.json`, `evaluation_result.json`, `eval_outputs/` — no `mutated_artifact_eval/` or `candidate_eval/`.
4. `mutated_artifact/` and root `mutated_artifact_eval/` are gone.
5. `eval_outputs/` contents are the same set of files as the old dump dir — only the directory name changed.
6. Repo grep for `baseline_eval`, `seed_candidate_eval`, `mutated_artifact_eval`, `candidate_eval`, `mutated_artifact`, `seed_candidate.py` returns zero live references (changelog/comments aside).
7. A short `--no-mutate` run produces `tries/try_001…NNN/`, each with the four expected entries minus `llm_response.json`.

If the scout findings contradict this plan — especially a second place the dump-dir name is derived — **stop and surface it before editing.**
