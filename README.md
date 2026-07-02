# Brief: eval-exposure experiment

Read SensorFlow primer first. Conventions: additive/minimal, fail loud (no try/except), reuse `_rng`, verify names/paths before coding. **Reality differs from brief → STOP, report. Never adapt silently.**

## Goal
New flag in `runner.py`: `--eval-exposure {none,summary,full}`, default `none`. Controls eval-code text injected into every mutation prompt:
- `none` → nothing. Prompts byte-identical to current. (control arm)
- `summary` → LLM-extracted contract, generated ONCE, cached to disk.
- `full` → entire evaluator source, verbatim.

## Forbidden
- No refactors. No new classes/config/plugins. One flag + one startup block + one prompt prepend + one log field.
- Eval context goes in PROMPT, never in `mutate()`'s `feedback=` arg. Don't touch feedback builders.
- Never regenerate summary per run.
- No new Bedrock client. Reuse the util `mutate()` already uses (find in `py_mutation_engine.py`). Can't find it → STOP.
- No try/except. Errors raise.

## Verify first
1. `runner.py`: argparse block, mutate() call sites + exact signature, where prompt built, where tries.csv row written.
2. `py_mutation_engine.py`: real name of Bedrock util. Below called `bedrock_complete` — replace.
3. Evaluator file path. Guess: `cached_evaluate_classifier_sensorflow.py`. Confirm.
4. `wandb_logger.py`: where run config set.

## Step 1 — flag
```python
parser.add_argument("--eval-exposure", choices=["none", "summary", "full"], default="none",
                    help="How much evaluator code the mutation LLM sees.")
```

## Step 2 — startup block (runner.py, module level)
```python
import hashlib
from pathlib import Path

EVALUATOR_PATH = Path("cached_evaluate_classifier_sensorflow.py")  # VERIFY
CONTRACT_PATH = Path("evaluation_contract.md")
CONTRACT_HASH_PATH = Path("evaluation_contract.hash")

CONTRACT_EXTRACTION_PROMPT = """Extract the candidate-facing contract from the evaluator code \
below. "Candidate" = Python file the evaluator loads and runs. Output ONLY these 4 sections, \
Markdown, nothing else. Telegraphic style, minimum words, under 45 lines total. But all \
identifiers, signatures, and metric-computation lines quoted VERBATIM — never paraphrase or \
rename an identifier.

1. Required interface: every name/function/signature candidate must define. Verbatim.
2. Input data: exact shapes/types/column names candidate receives. Verbatim names.
3. Metrics: for accuracy, f1_score_class1, depth, num_leaf_nodes — quote computing line(s), \
then <=1 terse sentence each + direction (max/min).
4. Failure conditions: how a candidate makes evaluator raise/error.

EXCLUDE: file IO, caching, logging, plotting, anything candidate doesn't see.

EVALUATOR CODE:
"""


def _evaluator_hash() -> str:
    return hashlib.sha256(EVALUATOR_PATH.read_bytes()).hexdigest()


def load_eval_context(mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "full":
        return EVALUATOR_PATH.read_text()
    current = _evaluator_hash()
    if CONTRACT_PATH.exists():
        if CONTRACT_HASH_PATH.read_text().strip() != current:
            raise RuntimeError(
                f"{EVALUATOR_PATH} changed since contract extracted. "
                f"Delete {CONTRACT_PATH} + {CONTRACT_HASH_PATH} to re-extract.")
        return CONTRACT_PATH.read_text()
    contract = bedrock_complete(CONTRACT_EXTRACTION_PROMPT + EVALUATOR_PATH.read_text())  # VERIFY util
    CONTRACT_PATH.write_text(contract)
    CONTRACT_HASH_PATH.write_text(current)
    print(f"--- extracted contract ---\n{contract}\n---")
    return contract
```
Call once before loop: `eval_context = load_eval_context(args.eval_exposure)`.
Bedrock util returns object not str → unwrap same way `mutate()` does.

## Step 3 — prompt prepend (all operations, identical)
```python
if eval_context:
    prompt = f"## Evaluation\n{eval_context}\nDefine the interface EXACTLY as named above.\n\n" + prompt
```
`none` mode ⇒ prompt byte-identical to current behavior.

## Step 4 — log arm
- `eval_exposure` column in every `tries.csv` row (constant per run — fine, self-labels files).
- `eval_exposure` in W&B config. Respect `--no-wandb`.
- Don't rename/reorder existing columns.

## Runbook (document, don't automate)
Same seed (module `_rng` already handles; create no new RNGs), same baseline, separate dirs:
```bash
python runner.py --eval-exposure none    --output-dir runs/arm1  # VERIFY flag name
python runner.py --eval-exposure summary --output-dir runs/arm2
python runner.py --eval-exposure full    --output-dir runs/arm3
```
Arm 2: before long run, grep each identifier in printed contract against evaluator file.

## Accept when
1. `none`: prompts byte-identical to pre-change.
2. `summary`: run 1 = exactly ONE extraction call, writes contract + hash; run 2 = zero calls, reads cache; evaluator edited → RuntimeError.
3. `full`: evaluator source verbatim in every mutation prompt.
4. `eval_exposure` in tries.csv + W&B config.
5. feedback/operators/RNG untouched; no try/except added.
6. Diff small, additive.
