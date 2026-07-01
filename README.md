
**Must-have — the Component 8 integration, since that's the only place I'm guessing against real code:**

1. **In `get_pareto.py`, the function that computes/returns the frontier** — its name, and what it actually returns. Specifically: a list of ids? What *type* are the ids (ints? strings? `try_000`-style zero-padded strings? bare `0`?) and are they `try_id`s or row indices? This drives the `normalize_id`/`normalize_frontier` contract — if they come back as `try_000` strings but the CSV `try_id` column is bare `0`, my normalization is wrong and every frontier border silently vanishes (the exact bug I've been guarding against). Paste that function's signature + its `return` line.

2. **The `pareto.png` render call site** — the few lines where `pareto.png` gets written, and *what calls that*. I need to see whether the frontier list is in scope right there (so the `render_lineage` call slots in beside it) or whether it's computed in a different function than the one that renders. Paste the render line plus its enclosing function header.

3. **How `tries.csv`'s path is known at that point** — is there a variable holding the run directory / csv path in scope at the render site? The recipe assumes `render_lineage` gets handed a path; I want to confirm one exists there rather than having to reconstruct it.

**High-value — confirms my data assumptions against the real file, not the screenshot:**

4. **The exact `tries.csv` header line, copy-pasted as text** (not a photo). The screenshot was enough to design against, but a text paste lets me confirm exact column spelling/spacing and settles whether `parent_try_ids` is truly the last column. `head -1 tries.csv`.

5. **Two real rows: the root and one combine** — `sed -n '2p;7p' tries.csv` or similar. Confirms the quoted-pair format survives a real CSV read and that the root's `parent_try_ids` is genuinely empty (not `"0"` or `"-1"` or `"None"` — any of which would change `parse_parents`).

**Nice-to-have — only if handy:**

6. **Whether `networkx` and/or `pandas` are already dependencies** of the project (`grep -i "networkx\|pandas" requirements.txt pyproject.toml` or just "is pandas imported in get_pareto.py"). This directly informs Fork A and Fork C — if pandas is already loaded everywhere and get_pareto hands you a DataFrame, Fork C's answer might flip to "just accept the DataFrame." If networkx is already in, Fork A's cost argument weakens.

7. **The Python version** (`python --version`) — only matters for whether I can use `X | None` syntax freely (3.10+) or should write `Optional[X]` (3.9 and under). Cosmetic but saves a revision.

If you get me **1, 2, 4, 5** I can lock the spine contracts with real confidence; **3, 6, 7** remove the last few assumptions. And **6** in particular feeds back into your A/B/C fork decisions — so it might be worth grabbing that one *before* you rule on the forks, since it could change my recommendation.

