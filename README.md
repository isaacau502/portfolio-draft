# SensorFlow — Budget-Based Stopping: Implementation Spec

**Status:** Ready to implement. Every product decision is resolved — do not re-decide them.
**Scope:** Add a dollar-spend stop condition to the search loop. **Metering + stop condition only.** No changes to operators, parent selection, evaluation, or Pareto-frontier logic.

**How this doc is written:** the implementing agent has the repository. So this spec does **not** describe the existing code's structure, function signatures, or where to splice things — read the repo for that. It gives two things only: (1) **new code, copy-ready** (§3–§5, §7) that stands alone, and (2) **behavioral requirements** (§6) stated as what must be true, which you satisfy however the existing code is shaped. Resolved decisions in §2 and out-of-scope items in §8 are fixed; do not "improve" them.

---

## 1. Feature summary

Today: `--tries N` runs N tries, then returns the Pareto frontier.

New: `--budget <dollars>` runs until accumulated AWS Bedrock spend reaches the budget, then returns the frontier. `--budget` and `--tries` are **either/or, whichever hits first**; both may be passed. **At least one of the two must be passed** — passing neither is a startup error (§5). When only `--budget` is given, `--tries` falls back to a runaway rail (§5/D6).

Spend is computed from the token counts Bedrock returns, priced by a small hardcoded table. The accumulator stores **tokens**; dollars are always **derived**, never stored as ground truth.

**Why budget, not just tries:** per-try cost is genuinely variable — Bedrock cost is dominated by output tokens, and reasoning/thinking depth varies per call. You can't reliably predict tries-per-dollar, so metering spend directly is the honest cap.

---

## 2. Resolved decisions (do not change)

| # | Decision |
|---|----------|
| D1 | Budget is **dollars**, user-facing. Internally accumulate raw tokens; price via the §3 table. |
| D2 | **Scalar accumulator** — one model per run, so one input-token total + one output-token total. No per-model buckets. |
| D3 | Rate resolved **once at run start** from the configured model id. Unknown model → **raise and refuse to start** (§3). Never fall back to a wrong rate. |
| D4 | `--budget` and `--tries` are **either/or, stop at whichever hits first.** **At least one must be passed** (passing neither is a startup error). When only `--budget` is given, `--tries` falls back to the rail in D6. |
| D5 | **Spend = money actually spent.** The accumulator updates the moment a Bedrock call returns usage. A try that calls Bedrock then fails downstream **still counts**. A try that fails **before** the Bedrock call counts **0 / 0**. |
| D6 | **Soft cap, no lookahead.** Run try 1 unconditionally; after each try add its cost, then check `spend >= budget` and stop if so. Overshoot by up to one try's cost is **expected, not a bug.** When `--budget` is given without `--tries`, `--tries` falls back to **300** (≈3× the typical 100-try run); this rail is the sole infinite-loop guard. |
| D7 | **No zero-cost canary** (considered, dropped). Runaway protection is the `--tries` rail in D4/D6 — it is therefore **load-bearing**, not a backup. |
| D8 | **Telemetry:** `tries.csv` gains exactly two columns, `input_tokens` + `output_tokens` (**no dollar column**). W&B logs cumulative `cost_so_far` per step. Run-end summary prints the dollar story once. Dollars appear only in the W&B curve and the summary. |

---

## 3. New module: pricing (copy-ready)

Rates are **USD per 1,000,000 tokens**, on-demand, standard region. **Verified June 2026 — re-verify before shipping at https://aws.amazon.com/bedrock/pricing/.**

```python
# pricing.py
PRICES_AS_OF = "2026-06"  # VERIFY against AWS Bedrock pricing before shipping.

# USD per 1,000,000 tokens (on-demand, standard region).
# Keyed by model family; matched via substring so version drift
# (e.g. opus-4-5 vs opus-4-7) does not break the lookup.
_PRICE_TABLE = {
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus":   {"input": 5.00, "output": 25.00},
}


class UnknownModelError(ValueError):
    """Configured model has no entry in the price table."""


def model_family(model_id: str) -> str:
    """Map a Bedrock model id to a price-table family. Single source of
    truth for the Sonnet/Opus decision, used by both resolve_rates and the
    run-end summary so pricing and display can never disagree. Raises
    UnknownModelError on anything unpriced (D3)."""
    m = model_id.lower()
    if "sonnet" in m:
        return "sonnet"
    if "opus" in m:
        return "opus"
    raise UnknownModelError(
        f"Model {model_id!r} is not in the price table "
        f"(known families: {sorted(_PRICE_TABLE)}). "
        f"Add a row to pricing._PRICE_TABLE and re-verify rates."
    )


def resolve_rates(model_id: str):
    """Return (input_rate, output_rate) in USD per 1M tokens for the
    configured model. Refuses unpriced models so the run fails before
    mis-pricing rather than silently using a wrong rate (D3)."""
    rates = _PRICE_TABLE[model_family(model_id)]
    return rates["input"], rates["output"]
```

> **Opus 4.7+ tokenizer note:** newer Opus may consume up to ~35% more tokens for the same text. The **dollars-per-token rate is unchanged** — only token counts rise, and those are measured live from the Bedrock response, so the substring family match stays correct. No special handling needed. (If a non-Sonnet/Opus model — e.g. Haiku — ever becomes runnable, add a row; until then, unknown models correctly fail loud.)

---

## 4. New module: cost accumulator (copy-ready)

```python
# cost.py
from dataclasses import dataclass
from typing import Optional
from pricing import resolve_rates


@dataclass
class CallUsage:
    """Token usage for a single Bedrock call, read from the response's
    usage metadata."""
    input_tokens: int
    output_tokens: int


class CostTracker:
    """Scalar token accumulator for one run (D2). Dollars are derived,
    never stored (D8). Construct once at run start from the configured
    model id; the constructor raises on an unpriced model (D3), so an
    unknown model fails the run before any Bedrock spend occurs."""

    def __init__(self, model_id: str):
        self.input_rate, self.output_rate = resolve_rates(model_id)
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, usage: Optional[CallUsage]) -> None:
        """usage is None when Bedrock was not called for this try (D5),
        which contributes nothing."""
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def dollars(self) -> float:
        return (self.input_tokens * self.input_rate
                + self.output_tokens * self.output_rate) / 1_000_000
```

---

## 5. CLI flags (copy-ready)

Add `--budget`. Give `--tries` a default of `None` so the "neither passed" case is detectable; the rail is applied at validation time.

```python
parser.add_argument(
    "--budget", type=float, default=None,
    help="Stop when accumulated Bedrock spend reaches this many US dollars. "
         "Combined with --tries, the run stops at whichever limit is reached "
         "first. Spend is estimated from token counts at rates as of "
         f"{PRICES_AS_OF}; see pricing.py.",
)
parser.add_argument(
    "--tries", type=int, default=None,
    help="Maximum number of tries, and the runaway guard when --budget is set "
         "(the loop cannot run longer than this even if the budget is never "
         "reached). If --budget is given without --tries, --tries falls back "
         "to 300. At least one of --budget or --tries must be provided.",
)
```

Startup validation (before the loop). After this block `args.tries` is always a valid int:

```python
TRIES_RAIL_DEFAULT = 300   # runaway guard when only --budget is given (D6)

if args.budget is None and args.tries is None:
    parser.error("provide --budget, --tries, or both")
if args.tries is None:                 # budget-only run: apply the rail
    args.tries = TRIES_RAIL_DEFAULT
if args.tries < 1:
    parser.error("--tries must be >= 1")
if args.budget is not None and args.budget <= 0:
    parser.error("--budget must be > 0")
```

---

## 6. Behavioral requirements (satisfy these against the existing loop)

These describe what must be true. Integrate them into the existing run loop however it is structured.

**6.1 Usage surfacing.** The mutation call (the only metered Bedrock call per try) must surface that call's input- and output-token counts to the loop, as a `CallUsage`. The counts come from the Bedrock response's usage metadata. If the mutation fails *before* the Bedrock call happens, the loop must treat usage as `None`.

**6.2 Tracker lifecycle.** Construct one `CostTracker(configured_model_id)` at run start, before the loop. This also performs the D3 unknown-model check up front.

**6.3 Charge timing (D5).** Commit the charge with a single `tracker.add(usage)` at the moment the Bedrock response is parsed — i.e. inside the per-try work, right after the mutation call returns and **before evaluation runs**. Because `tracker` is shared run state mutated in place, the spend is recorded the instant it is incurred, even if a later step in the same try fails. Call `add` **exactly once per try**; do **not** also add in the loop body — that double-charges. Within-try mutation/evaluation failures must be **recorded as results, not propagated** out of the try, so control always returns to the loop for the post-try budget check (6.4); this matches the existing archive design, which records both successes and failures.

**6.4 Termination (D4/D6).** The loop must:
- run try 1 unconditionally;
- after each try is fully completed (mutated, evaluated, recorded), check the limits;
- stop when **either** `args.budget is not None and tracker.dollars() >= args.budget` **or** the try count reaches `args.tries` — whichever happens first;
- keep `try_id` starting at 1 and incrementing by 1, as today.

The budget check is **post-try** (after the spend is incurred and recorded), giving the intended soft cap with overshoot ≤ one try. Record which limit stopped the run (`"budget"` vs `"tries"`) for the summary; treat "hit the default 300 rail" and "hit a user-set `--tries`" identically as `"tries"`.

The new control flow, in isolation (adapt variable/call names to the repo):

```python
tracker = CostTracker(configured_model_id)   # 6.2; raises on unknown model

try_id = 1
stop_reason = "tries"
while try_id <= args.tries:
    # Existing per-try work: mutate -> evaluate -> record (archive, csv, wandb).
    # Inside it, the charge is committed exactly once at the Bedrock boundary
    # (6.3); `usage` is surfaced only so the CSV row can record token counts
    # (6.5). Within-try failures are recorded, not raised, so control always
    # returns here for the budget check below.
    usage = run_one_try(try_id)              # CallUsage, or None if Bedrock was not called
    # Do NOT call tracker.add(usage) here — already committed inside the try (6.3).
    if args.budget is not None and tracker.dollars() >= args.budget:
        stop_reason = "budget"
        break
    try_id += 1

tries_completed = try_id if stop_reason == "budget" else try_id - 1
```

> `tries_completed` differs by branch on purpose: on a budget break the current `try_id` did run, so it counts; on tries exhaustion the loop has incremented past the last completed try, so subtract one.

**6.5 `tries.csv` (D8).** Append exactly two columns, `input_tokens` and `output_tokens`, to the existing per-try rows (append at the end so name-indexed parsers are unaffected). Write the try's token counts; for a try with no Bedrock call (`usage is None`), write `0` and `0`. Write **no** dollar/cost column — the dollar figure stays recomputable from these token columns plus `pricing.py`, so a wrong rate can never corrupt a stored record.

**6.6 W&B (D8).** Once per try, at `step=try_id`, guarded by the existing `--no-wandb` flag, log the cumulative spend:

```python
if not args.no_wandb:
    wandb.log({"cost_so_far": tracker.dollars()}, step=try_id)
```

`cost_so_far` is cumulative and non-decreasing → it renders as the spend curve. Add nothing else to W&B. Under `--no-wandb`, log nothing.

**6.7 Run-end summary.** After the loop, print one summary block (copy-ready; adjust wording, keep the fields):

```python
from pricing import PRICES_AS_OF, model_family

def print_run_summary(tracker, tries_completed, stop_reason, model_id):
    family = model_family(model_id).title()   # "Sonnet"/"Opus"; same source as pricing (won't
                                              # raise here — model was validated at run start)
    reason = "budget reached" if stop_reason == "budget" else "tries limit reached"
    print(
        f"\nRun complete: {tries_completed} tries, "
        f"~${tracker.dollars():.2f} on {family}.\n"
        f"  Stopped: {reason}.\n"
        f"  Tokens: {tracker.input_tokens:,} in / {tracker.output_tokens:,} out.\n"
        f"  Rates: ${tracker.input_rate:.2f}/1M in, "
        f"${tracker.output_rate:.2f}/1M out (as of {PRICES_AS_OF} — estimate).\n"
    )
```

Required fields: tries completed, total spend (derived), model, which limit stopped the run, total in/out tokens, the assumed rates, and `PRICES_AS_OF`.

---

## 7. Acceptance tests

Implement these (pytest names suggested). Stub the mutation call to return a `CallUsage(input_tokens=…, output_tokens=…)` (or `None`) so tests need no live Bedrock call.

| Test | Setup | Assert |
|------|-------|--------|
| `test_budget_only_stops_on_spend` | `--budget 20`, fixed known cost per try | Stops at the first try where cumulative `dollars() >= 20`; overshoot ≤ one try; frontier returned. |
| `test_whichever_first_tries_wins` | `--budget 1000 --tries 5`, cheap stub | Exactly 5 tries; `stop_reason == "tries"`. |
| `test_whichever_first_budget_wins` | `--budget 1 --tries 500`, costs crossing $1 at try 3 | Stops at try 3; `stop_reason == "budget"`. |
| `test_no_flags_rejected` | neither flag | Startup `parser.error` ("provide --budget, --tries, or both"); no try runs. |
| `test_budget_only_applies_tries_rail` | `--budget` only (large), cheap stub | Loop cannot exceed 300 tries even though budget is never reached. |
| `test_unknown_model_refuses_to_start` | configured model id containing neither "sonnet" nor "opus" | `CostTracker(...)` raises `UnknownModelError`; no try runs. |
| `test_failed_after_bedrock_still_charged` | stub returns valid `CallUsage`, then force eval failure | Tokens counted in tracker and written to CSV; `dollars()` reflects them. |
| `test_failed_before_bedrock_not_charged` | stub yields `usage is None` | Contributes `0/0`; `dollars()` unchanged; CSV row is `0,0`. |
| `test_csv_has_token_columns_no_dollars` | any short run | `tries.csv` header has `input_tokens` and `output_tokens` and **no** dollar column. |
| `test_wandb_cost_curve_monotonic` | W&B mocked, multi-try run | `cost_so_far` logged once per `try_id`, non-decreasing. |
| `test_no_wandb_logs_nothing` | `--no-wandb` | No W&B calls made. |
| `test_summary_fields_present` | any run | Summary contains tries completed, spend, model, stop reason, in/out tokens, rates, `PRICES_AS_OF`. |
| `test_tries_zero_rejected` | `--tries 0` | Startup `parser.error`. |
| `test_budget_nonpositive_rejected` | `--budget 0` | Startup `parser.error`. |

---

## 8. Out of scope (do not add)

- **Live pricing** (AWS Price List API / `boto3.client('pricing')`). Rejected: adds an external dependency + model-name-mapping liability to a feature whose job is to *not* run unbounded. The two hardcoded rows are the intended design.
- Per-token optimization or prompt-shortening.
- Mid-run budget reallocation across operators.
- Pre-try cost prediction / lookahead (D6 is a post-pay soft cap by design).
- Any refactor of operators, parent selection, evaluation, archive, or frontier logic.
- A `cost_so_far` CSV column, or any stored dollar value (D8).
- Re-introducing the zero-cost canary (D7).
