# SensorFlow Task Brief — Model-Agnostic ROM/RAM Complexity Proxies

> **Pair this with the SensorFlow Project Context Primer.** The primer explains the project,
> archive/metric data shapes, and working conventions. This brief says what to build. Where they
> conflict, the primer's conventions win.

---

## 1. Goal

Replace tree-specific complexity metrics as the *search objectives* with two **model-agnostic**
memory proxies, so the evolutionary search can rank candidates from ANY joblib-loadable classifier
family (single trees, forests, gradient boosting, logistic regression, MLPs, XGBoost, LightGBM)
on a common scale:

- **`rom_bytes`** (minimize) — proxy for flash/storage footprint of the fitted model.
- **`ram_bytes`** (minimize) — proxy for peak working memory during a single inference.

These are **ranking proxies for automated search**, NOT hardware-accurate estimates. Eventual
deployment is C firmware; when that phase arrives, only the proxy functions get swapped —
metric keys and directions must stay stable from day one.

## 2. Decisions already made (do not relitigate)

1. **Pickle/joblib blob size is REJECTED as the ROM proxy.** Empirical reason: a bare fitted
   `DecisionTreeClassifier` pickles to ~2–3 KB of framing/metadata before any nodes, while the
   actual payload of a ~20-node tree (the ballpark model size for this project) is ~1.4 KB. The
   signal (15 vs 25 nodes ≈ a few hundred bytes) drowns in a family-dependent constant, which
   both flattens within-family ranking and skews cross-family ranking. Do not use serialized
   size anywhere in the metric path.
2. **Scalar-counting at a firmware-ish cost model is the ROM proxy.** Count learned parameters /
   tree nodes via structural introspection and cost them at 4 bytes per scalar (float32), with
   ~4 scalars per tree node (feature index, threshold, two child references) plus per-leaf output
   values. Absolute numbers are approximations; only ordering matters.
3. **Detection is duck-typed on fitted attributes, not an isinstance enumeration.** sklearn
   convention: learned state lives in trailing-underscore attributes (`tree_`, `estimators_`,
   `coef_`, `coefs_`). This makes most present and future sklearn families work with zero new
   code (any ensemble recurses through `estimators_`; any linear model hits `coef_`). Only the
   C-backed boosting libraries (XGBoost, LightGBM) need explicit branches.
4. **Unknown model types RAISE.** No silent fallback ROM estimate. Per project convention
   ("anything that can't happen in a correct run just raises"), a mutated candidate producing an
   estimator the proxy doesn't understand must fail evaluation loudly, not receive a garbage
   complexity score.
5. **RAM is included now even though it is near-constant for current candidates.** With ~4–6
   input features, trees/forests/linear models have negligible working memory; only MLPs vary.
   It ships anyway so the metric key exists in `tries.csv` and W&B from day one (schema
   stability), and it is total by construction (never raises).
6. **Both metrics enter the metric policy** (direction: min). Multi-objective Pareto handling is
   the project's mechanism — no weighted combination of ROM and RAM into one score.
7. **Existing keys `depth` and `num_leaf_nodes` are NOT renamed or removed** in this task. Open
   follow-up decision for the maintainer (do not implement): whether to drop them from the
   *policy* (keeping them in metrics) once ROM subsumes them, since four size-flavored
   objectives inflate the frontier with redundant non-dominated points. Flag it; don't do it.

## 3. Reference implementation

Adapt names/style to the codebase; behavior below is the spec. Assume binary classification
(`n_classes = 2`) and get `n_features` from the trained model or the evaluator's feature list —
do not hardcode 6 if the true count is available at the call site.

```python
import numpy as np

BYTES_PER_SCALAR = 4  # float32 firmware assumption


def rom_bytes(model) -> int:
    """Approximate flash footprint of a fitted classifier, in bytes.

    Ranking proxy only. Duck-typed dispatch; raises on unknown families.
    Order of checks matters: ensembles are checked via estimators_ and
    recurse into this function for their members.
    """
    if hasattr(model, "tree_"):  # single decision tree
        t = model.tree_
        # per node: feature idx + threshold + 2 child refs ~= 4 scalars,
        # plus the per-leaf class-value array
        return t.node_count * 4 * BYTES_PER_SCALAR + t.value.size * BYTES_PER_SCALAR

    if hasattr(model, "estimators_"):  # RandomForest, GradientBoosting, Bagging, ...
        # GradientBoosting stores a 2-D array of trees; ravel handles both shapes.
        return int(sum(rom_bytes(e) for e in np.ravel(model.estimators_)))

    if hasattr(model, "coefs_"):  # sklearn MLPClassifier
        return BYTES_PER_SCALAR * (
            sum(w.size for w in model.coefs_)
            + sum(b.size for b in model.intercepts_)
        )

    if hasattr(model, "coef_"):  # LogisticRegression, linear SVM, SGDClassifier, ...
        return BYTES_PER_SCALAR * (model.coef_.size + np.asarray(model.intercept_).size)

    if hasattr(model, "get_booster"):  # XGBoost sklearn wrapper
        n_nodes = len(model.get_booster().trees_to_dataframe())
        return n_nodes * 4 * BYTES_PER_SCALAR

    if hasattr(model, "booster_"):  # LightGBM sklearn wrapper
        dump = model.booster_.dump_model()
        n_nodes = sum(_lgbm_tree_nodes(t["tree_structure"]) for t in dump["tree_info"])
        return n_nodes * 4 * BYTES_PER_SCALAR

    raise TypeError(f"rom_bytes: no ROM model for {type(model).__name__}")


def _lgbm_tree_nodes(node: dict) -> int:
    """Count nodes in a LightGBM dumped tree structure (leaves + splits)."""
    if "leaf_index" in node or "leaf_value" in node and "left_child" not in node:
        return 1
    return 1 + _lgbm_tree_nodes(node["left_child"]) + _lgbm_tree_nodes(node["right_child"])


def ram_bytes(model, n_features: int, n_classes: int = 2) -> int:
    """Approximate peak working memory for one inference, in bytes.

    Total by construction — never raises. Near-constant until MLP
    candidates appear; included for schema stability.
    """
    base = n_features * BYTES_PER_SCALAR  # input feature vector

    if hasattr(model, "coefs_"):  # MLP: two largest adjacent activation buffers
        widths = [n_features, *model.hidden_layer_sizes, n_classes]
        peak = max(widths[i] + widths[i + 1] for i in range(len(widths) - 1))
        return base + peak * BYTES_PER_SCALAR

    if hasattr(model, "estimators_") or hasattr(model, "get_booster") or hasattr(model, "booster_"):
        return base + n_classes * BYTES_PER_SCALAR  # vote / margin accumulator

    return base  # single tree, linear: O(1) traversal state
```

Implementation notes:

- `np.ravel(model.estimators_)` is deliberate: `GradientBoostingClassifier.estimators_` is a 2-D
  ndarray of regressors, `RandomForestClassifier.estimators_` is a flat list. Ravel covers both.
- The XGBoost `trees_to_dataframe()` call costs a few ms (pandas construction). Acceptable — see
  performance section. If it ever matters, count from `get_booster().get_dump()` instead.
- The LightGBM leaf-detection in `_lgbm_tree_nodes` should be verified against an actual
  `dump_model()` output when a LightGBM candidate first appears; the dump schema uses
  `left_child`/`right_child` dicts for splits and `leaf_value` leaves. If the schema check fails
  in practice, fix the walker — do not silently return 0.
- Do not add defensive try/except around dispatch. Raising on unknown types is the spec.

## 4. Integration points (verify against real code first — line numbers drift)

Per the primer's feasibility-first rule: confirm each anchor below by function/name search before
editing. If reality differs, STOP and report.

1. **Compute the metrics** in `cached_evaluate_classifier_sensorflow.py`, immediately after the
   model is fitted (the same place confusion-matrix counts are populated). Add to the metrics
   dict: `"rom_bytes": rom_bytes(model)`, `"ram_bytes": ram_bytes(model, n_features)`. Get
   `n_features` from the evaluator's actual feature list at that call site.
2. **Register objectives** in the metric policy in `runner.py` (~line 45; anchor to the policy
   dict by its existing keys `accuracy`, `f1_score_class1`, `depth`, `num_leaf_nodes`). Add
   `rom_bytes` → min and `ram_bytes` → min. Do not touch existing entries.
3. **Where the functions live:** simplest placement consistent with the codebase — e.g. directly
   in `cached_evaluate_classifier_sensorflow.py` or a small sibling module. Do not build a
   plugin/registry system; two functions and one private helper is the whole surface.
4. **No changes** to `mutate()`, parent selection, `choose_operation`, the archive shape, or
   `pareto_frontier_try_ids` — the frontier code consumes whatever keys the policy declares.
5. **Logging follows automatically**: `tries.csv` and W&B log the metrics dict; verify the new
   keys appear in one smoke-run row rather than assuming.
6. Remember the archive data shape (primer): `entry["result"]` is an `EvaluationResult` OBJECT —
   attributes `.status`, `.metrics`, `.failure_reason` — not a dict. Don't index it.

## 5. Improve-operator note

The `improve` operation cycles through metric-policy keys as targets. Adding two keys means the
LLM will eventually be asked to "improve rom_bytes" / "improve ram_bytes". Check the improve
feedback builder: if it phrases targets generically from the policy (metric name + direction),
nothing is needed. If it has per-metric wording, add one line of phrasing for each new key
(e.g. "reduce the model's stored size" / "reduce inference working memory"). Do not restructure
the feedback builder.

## 6. Performance (already analyzed — do not add caching)

All proxies are attribute reads plus trivial arithmetic: sub-microsecond for trees/linear/MLP,
tens of microseconds for a forest loop, a few milliseconds worst-case for the XGBoost dataframe
branch. Per-try cost is dominated by candidate training, dataset evaluation, and the Bedrock
mutation call; these functions add well under 0.1%. No caching, no benchmarking.

## 7. Acceptance checklist

- [ ] A run over existing tree candidates completes; every successful try's metrics include
      `rom_bytes` and `ram_bytes`; a ~20-node tree lands in the right ballpark
      (~360 B ROM = 20·4·4 + 20·2·4 for a binary tree; exact value depends on `value.size`).
- [ ] 15-node vs 25-node trees show a clearly separated `rom_bytes` (~40% relative gap).
- [ ] `rom_bytes` on a fitted `RandomForestClassifier`, `LogisticRegression`, and
      `MLPClassifier` returns plausible, family-consistent values without any isinstance on
      those classes.
- [ ] An unfitted or unknown-type object raises `TypeError` (or `AttributeError` for unfitted —
      loud failure either way is acceptable).
- [ ] `ram_bytes` never raises; MLP with hidden layers returns more than a tree.
- [ ] New keys appear in `tries.csv` and W&B; Pareto frontier computation runs with the extended
      policy; `pareto.png` renders.
- [ ] `depth` / `num_leaf_nodes` untouched; no existing key renamed; no RNG touched; diff is
      additive and small.

## 8. Out of scope

- Hardware-accurate memory estimation, codegen sizing, quantization assumptions beyond float32.
- Removing `depth`/`num_leaf_nodes` from the policy (flag as follow-up only).
- Ops-per-inference / latency metric (discussed, deliberately deferred).
- Any evaluator interface generalization/validation, weighted scalarization, plugin registry.
