# SensorFlow — W&B Logger Improvements: Build Recipe (paste-by-paste)

Paste CHUNK 0 first (scout). Then CHUNK 1, then CHUNK 2, verifying each before the next. Each step
assumes CHUNK 0's findings are in the session.

---

## CHUNK 0 — Scout the logger (Haiku; read-only, no code)

**Read-only. Don't change anything. Report these, with file:line + pasted lines:**

1. `make_logger(args)` (~runner.py:494) — its exact signature, and how it constructs the
   `WandbLogger` (what it passes in).
2. `WandbLogger.__init__` and `WandbLogger.log_evaluation` in `wandb_logger.py` — exact signatures.
   (Scout already found `log_evaluation(result, step=step)`.)
3. `evaluation_result_to_wandb_dict(result)` in `wandb_logger.py` — paste the function body (it
   currently hardcodes only `accuracy`).
4. The `log_evaluation` guard in `runner.py` (the None-safe/exception-swallowed wrapper that
   delegates to `logger.log_evaluation`) — paste it, and the per-try call site where it's called
   (the one with `step=try_id`). Report what variables are in scope there — specifically, is `op`
   (the operation string) available at that call site?
5. Where `METRIC_POLICY` is defined (~runner.py:45) and confirm it's importable / passable into the
   logger.
6. Confirm whether `wandb.config` is set anywhere today, and where `wandb.init` is called.

**Report as 6 short answers. No fixing.** End with one line: are `make_logger` /
`WandbLogger.__init__` / `log_evaluation` / the guard the only signatures we'd touch, or are there
others?

---

## CHUNK 1 — Make the logger policy-aware + log everything (Sonnet)

**Model:** Sonnet — several coherent edits in one file (`wandb_logger.py`), best done with the whole
file in view in one pass.

**Goal:** the logger logs all numeric metrics (not just accuracy), tracks best-per-objective using
the metric policy, and records run config. All internal to `wandb_logger.py` + the logger's init.

**Do:**

1. **Generalize `evaluation_result_to_wandb_dict`** — replace the hardcoded `accuracy` block with a
   loop over all numeric metrics:
   ```python
   metrics = _get_attr(result, "metrics")   # however it's currently read; EvaluationResult attr
   if metrics is not None:
       for k, v in metrics.items():
           if _is_number(v):
               out[f"metrics/{k}"] = float(v)
   ```
   Keep the existing `candidate/status_ok` / `candidate/status_error` fields exactly as they are.
   This logs the four policy metrics AND the `nb_segments_*` diagnostics — that's fine, leave them.

2. **Thread `metric_policy` into the logger.** Add a `metric_policy: dict` param to
   `WandbLogger.__init__` (and pass it from `make_logger`, which has `args`/policy access — confirm
   from scout). Store it on the instance.

3. **Emit `define_metric` from the policy, once at init** (after `wandb.init`):
   ```python
   for name, direction in self.metric_policy.items():
       summary = "max" if direction == "max" else "min"
       wandb.define_metric(f"metrics/{name}", summary=summary)
   ```
   Only policy metrics get best-tracking. Do NOT define_metric the `nb_segments_*` diagnostics
   (they have no better/worse direction). The logger has no prior concept of metric_policy — this
   is a new addition, not a tweak.

4. **Dump run config at init** — set `wandb.config` (or pass `config=` to `wandb.init`) with the run
   params: tries, `REPAIR_PROB`, `COMBINE_PROB`, `COMBINE_MIN_FRONTIER`, dataset/baseline path, and
   `metric_policy`. Log everything available at init; we'll prune later. Pull these from whatever
   `make_logger` already receives (`args`); pass through as needed.

**Don't:**
- Don't change `log_evaluation`'s signature here — that's CHUNK 2.
- Don't hardcode any metric name (no `accuracy`, no policy keys typed out — read them from
  `metric_policy` / the metrics dict).
- Don't define_metric the diagnostics. Don't drop the existing status fields.
- Don't add a custom x-axis (`step=try_id` is already the x).

**Verify:** run 5–10 tries with W&B on. In the dashboard confirm: (1) four `metrics/*` series plotted
over try_id, not one; (2) the run summary shows best-per-objective respecting direction (max for
accuracy/f1, min for depth/leaf) — not the last try's value; (3) the run config shows the params.

**Feasibility first:** confirm from CHUNK 0 that `make_logger` can supply `metric_policy` + the
config params to `WandbLogger.__init__`. If `make_logger` doesn't currently receive the policy/args
needed, report how it's reachable before editing. Report the `_get_attr`/`_is_number` helpers exist
(the scout saw them) so the loop reuses them.

---

## CHUNK 2 — Log operation context per try (Sonnet)

**Model:** Sonnet — one logical change threaded across the call chain (runner call site → guard →
`log_evaluation`); keep it consistent end-to-end in one pass.

**Goal:** each try logs which operation fired, so W&B can chart/filter op frequency over time.

**Do:**

1. **Add an `extra: dict | None = None` param** to `WandbLogger.log_evaluation`; merge `extra` into
   the payload dict before `wandb.log` (`payload.update(extra or {})`).

2. **Thread `extra` through the `runner.py` guard** — the None-safe wrapper that delegates to
   `logger.log_evaluation`: add the same `extra=None` param and forward it.

3. **At the per-try call site** (where `op` is in scope — confirm from scout), build the op dict and
   pass it as `extra`:
   ```python
   op_extra = {
       "operation": op,                       # string, for readable filtering
       "operation/repair":  1 if op == "repair"  else 0,   # one-hot, for charting
       "operation/improve": 1 if op == "improve" else 0,
       "operation/combine": 1 if op == "combine" else 0,
   }
   # pass op_extra as extra= into the guard's log_evaluation call (step=try_id unchanged)
   ```
   One-hot columns let W&B chart op frequency / stack them over try index; the string supports
   grouping/filtering.

4. **Baseline call site** (`step=0`) has no operation — pass `extra=None` or omit; don't invent an op for it.

**Don't:**
- Don't change `step=try_id` / `step=0`.
- Don't log the op as a second separate `wandb.log` call — route it through `extra` so it's one
  payload per step (avoids split-logging drift).
- Don't touch the metrics/config logic from CHUNK 1.

**Verify:** run 8–10 tries. In W&B confirm per-try rows have `operation` (string) and the three
`operation/*` one-hot columns; confirm you can make a stacked/area chart of op counts over try_id,
and group runs by `operation`. Baseline (step 0) has no op columns or they're absent — fine.

**Feasibility first:** confirm `op` is in scope at the per-try `log_evaluation` call site (from
CHUNK 0). If it isn't, STOP and report where the operation string lives at that point — do not
recompute it.

---

### Order recap
`0 scout (Haiku) → 1 logger internals: metrics loop + define_metric from policy + config (Sonnet)
→ 2 operation context via extra param, threaded runner→guard→logger (Sonnet)`. Chunk 1 and 2 are
independent (1 = what's logged about the result, 2 = op context); do 1 first since it's the
higher-value demo win. Each chunk: confirm feasibility → implement → verify in the W&B dashboard.
