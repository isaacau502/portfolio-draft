CHUNK 0 — Context (paste first, don't implement anything)

You are building lineage_viz.py for SensorFlow, an LLM-guided
evolutionary search that optimizes tiny ML candidates (decision trees for
car-crash detection). The search records every try in a flat CSV; the Pareto
frontier is the implicit quality filter. You are adding one new, self-contained
module that reads that CSV and renders a lineage graph (a DAG of which try
spawned which, via which operation) as a PNG/SVG using graphviz. Nodes = tries,
edges = the operation that produced each child, colored by operation; node
border = role (frontier / dominated / error); node fill = a muted-blue temporal
ramp by try order.

Key facts you must rely on (don't re-derive):


Data source is tries.csv. Relevant columns: try_id (bare int),
parent_try_ids (LAST column — comma-joined parents, e.g. "0,3" for a
combine; empty for the root; this is the edge source), operation
(improve/combine/repair; empty for root), status (ok/error),
f1_score_class1, num_leaf_nodes, accuracy, depth, failure_reason.
There is a singular parent_try_id column too — ignore it entirely;
it drops the 2nd parent of every combine. Edges come ONLY from
parent_try_ids (plural).
The frontier is produced by pareto_frontier_try_ids(rows, metric_policy) in
get_pareto.py, which returns a list of bare int try_ids. Do not modify it.
The render integration point is inside plot_pareto(...) in get_pareto.py,
right after it writes pareto.png (that function computes a frontier_ids
set and holds csv_path and out_path locals at that point). Only CHUNK 10
touches that file.
Environment: dot binary on PATH (graphviz installed), the pip graphviz
package, Python 3.10+. No pandas, no networkx in the project.
Full pinned signatures + contracts live in Step 3 of this recipe; the build
order in Step 4. This script implements them in dependency order.


Global rules for every step (guardrails — obey in all chunks):


Build the exact function names + signatures given in each chunk. Do not
rename them, and do not rename existing repo symbols (plot_pareto,
pareto_frontier_try_ids, frontier_ids, csv_path, out_path, CSV columns).
Additive only. Everything goes in the new lineage_viz.py. The ONLY edit
to existing code is the single insertion in CHUNK 10. Touch no search logic,
loop, evaluator, or the body of plot_pareto beyond that one line.
No new dependencies — stdlib csv + graphviz only.
Normalize all ids to str on both CSV and frontier sides (frontier
arrives as bare ints; without this the frontier borders silently vanish).
Determinism: insert nodes/edges in ascending int(try_id) order.
No W&B. No scope creep — no CLI, no config system, no output formats beyond
png/svg-inferred-from-extension, no interactive/replay, no extra node channels.
You may decide local details (private helper internals, number formatting
within the stated rules). You may NOT change contracts, control flow, or any
"Don't" listed.
Feasibility first (every chunk): before writing code, confirm the chunk is
feasible against what already exists — for new-file chunks, that the functions
this chunk calls were built in prior chunks with the stated signatures; for
CHUNK 10, against the real get_pareto.py. If anything differs from what the
chunk claims, STOP and report — do not silently adapt. When ambiguous, ask one
question rather than guess.


Acknowledge you've read this and wait for CHUNK 1. Do not write code yet.

Notes that apply to all Verify steps:


No automated test suite — verification is by eyeball / a quick throwaway
snippet in a REPL or if __name__ == "__main__" scratch. Do not create a
test_*.py file or a pytest harness. Just confirm the stated behavior holds,
report it, and move on.
"read g.source" means the string returned by graphviz.Digraph.source;
"contains X" checks mean substring presence (e.g. "penwidth=3.0" in g.source),
not exact-line equality — graphviz's attribute ordering/quoting is not
guaranteed.
The sample run for any Verify step that needs real data is
runs/run_test_opfirst_mutate/ — its tries.csv is the 26-row run
referenced throughout. Read from runs/run_test_opfirst_mutate/tries.csv. For
any output file you write during verification, pick a scratch path yourself
(e.g. a temp dir or the run dir); it doesn't matter where.



CHUNK 1 — Foundations: constants, TryRecord, id plumbing

Model: Haiku.  Depends on: CHUNK 0.

Goal: lay down the module's constants, the row dataclass, and all id/string
normalization — the leaf layer everything else builds on.

Do:


Create lineage_viz.py starting with this exact import block:


python  import csv
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Iterable, Literal
  import graphviz


Then module constants:
RAMP_START = "#EAF2FB", RAMP_END = "#5B7FA6",
EDGE_COLORS = {"improve": "#E69F00", "combine": "#CC79A7", "repair": "#D55E00"},
UNKNOWN_EDGE_COLOR = "#666666",
BORDER = {"frontier": {"penwidth":"3.0","color":"#222222"}, "dominated": {"penwidth":"1.0","color":"#999999"}, "error": {"penwidth":"1.5","color":"#999999"}},
and Role = Literal["frontier","dominated","error"].
Define @dataclass class TryRecord with fields: try_id: str,
parent_ids: list[str], operation: str, status: str,
f1_score_class1: float | None, num_leaf_nodes: int | None,
accuracy: float | None, depth: int | None, failure_reason: str | None.
Implement:

normalize_id(raw: int | str | None) -> str — str(raw).strip(); None → "".
parse_parents(parent_try_ids_field: str | None) -> list[str] — split on
comma, normalize_id each, drop empties; ""/None → [].
normalize_frontier(frontier_try_ids: Iterable[int|str] | None) -> set[str] —
normalize_id each into a set; None → set().
is_frontier(try_id: str, frontier: set[str]) -> bool.





Don't:


Don't read or reference the singular parent_try_id column anywhere.
Don't rename the constants or dataclass fields.


Verify: normalize_id(5)=="5", ("  5 ")=="5", (None)=="";
parse_parents("0,3")==["0","3"], ("")==[]; normalize_frontier([0,5])=={"0","5"};
is_frontier("5",{"5"}) True.

Feasibility first: confirm the file doesn't already exist / is the empty
scaffold from CHUNK 0; report before coding.


CHUNK 2 — Temporal color ramp

Model: Haiku.  Depends on: CHUNK 1 (RAMP_* constants).

Goal: map a try_id to its muted-blue fill color, end to end.

Do: implement


hex_to_rgb(hex_color: str) -> tuple[int,int,int]
rgb_to_hex(rgb: tuple[int,int,int]) -> str
lerp_rgb(c0: tuple[int,int,int], c1: tuple[int,int,int], t: float) -> tuple[int,int,int] (clamp t to [0,1])
compute_id_range(try_ids: Iterable[str]) -> tuple[int,int] (min/max as ints)
normalize_try_id(try_id: str, id_min: int, id_max: int) -> float (guard: id_max==id_min → 0.0)
temporal_fill_color(try_id: str, id_min: int, id_max: int) -> str (ties the above; interpolates RAMP_START→RAMP_END)


Don't:


Don't introduce a colormap library; linear RGB interpolation only.
Don't use CSV depth for anything here (it's unrelated to color).


Verify: hex round-trips; lerp at t=0→c0, t=1→c1; normalize_try_id min→0.0,
max→1.0, zero-range→0.0; compute_id_range([]) raises ValueError;
temporal_fill_color("0",0,25)==RAMP_START, ("25",0,25)==RAMP_END.

Feasibility first: confirm RAMP_START/RAMP_END exist from CHUNK 1.


CHUNK 3 — Node presentation logic (pure)

Model: Haiku.  Depends on: CHUNK 1 (BORDER, TryRecord).

Goal: compute a node's role and its text/attrs — no graph object yet.

Do: implement


determine_role(status: str, on_frontier: bool) -> Role — "error" if
status=="error" (error wins); else "frontier" if on_frontier; else
"dominated".
border_attrs_for_role(role: Role) -> dict[str,str] — return {"penwidth","color"}
from BORDER[role]. Do NOT set style here (composition happens in CHUNK 6).
format_node_label(try_id: str, on_frontier: bool, f1: float|None, num_leaf_nodes: int|None) -> str —
non-frontier: return try_id alone. Frontier: join with literal "\n" the
lines try_id, then f"f1={f1:.3f}" only if f1 is not None, then
f"leaves={num_leaf_nodes}" only if num_leaf_nodes is not None (check
None BEFORE formatting so :.3f never hits None).
build_node_tooltip(record: TryRecord) -> str — join with literal "\n" these
exact key: value lines in this order, rendering any None as -:
try_id, operation, status, f1_score_class1, accuracy, depth,
num_leaf_nodes, parents (value = ",".join(record.parent_ids) or - if
empty). Append a final failure_reason: {…} line only if
record.status == "error". Use those literal key strings (e.g. f1_score_class1,
not f1).


Don't:


Don't put operation on the node (operation lives on edges, CHUNK 5).
Don't include style in the border dict.


Verify: determine_role("error",True)=="error", ("ok",True)=="frontier",
("ok",False)=="dominated"; border dict matches BORDER; frontier label is 3
lines with f1=0.498; tooltip has failure_reason only for errors.

Feasibility first: confirm BORDER and TryRecord exist from CHUNK 1.


CHUNK 4 — CSV loader

Model: Haiku.  Depends on: CHUNK 1 (TryRecord, normalize_id, parse_parents).

Goal: turn tries.csv into list[TryRecord].

Do: implement


private _to_float_or_none(s: str|None) -> float|None and
_to_int_or_none(s: str|None) -> int|None (blank/None → None).
load_tries(tries_csv_path: str | Path) -> list[TryRecord] — read with
csv.DictReader (handles the quoted "0,3" cell); per row normalize_id the
try_id, parse_parents the plural column, strip operation/status, coerce
numerics; return the list in file order.


Don't:


Don't hand-split raw lines on comma (the quoted combine cell needs a real CSV
reader).
Don't read the singular parent_try_id column.
Don't silently skip a missing expected column — raise a clear error naming it.


Verify: against the real tries.csv: 26 records; root (try 0)
parent_ids==[], operation==""; combine (try 5) parent_ids==["0","3"],
operation=="combine"; error (try 8) status=="error" with a failure_reason;
blank numerics → None; header-only file → [].

Feasibility first: read the header of runs/run_test_opfirst_mutate/tries.csv
and confirm it matches the columns in CHUNK 0 (spelling/order) and that
parent_try_ids is the last column. Report the header before coding; if it
differs, STOP and report.


CHUNK 5 — Graph decoration primitives

Model: Haiku.  Depends on: CHUNK 1 (EDGE_COLORS).

Goal: the small graphviz mutators + string helpers that don't need the node
styler.

Do: implement


edge_color_for_operation(operation: str) -> str — EDGE_COLORS lookup;
unknown → UNKNOWN_EDGE_COLOR (don't raise).
style_edge(g, child_id: str, parent_id: str, operation: str) -> None —
g.edge(parent_id, child_id, color=edge_color_for_operation(operation)).
apply_graph_attrs(g, caption: str) -> None — g.attr(rankdir="TB", label=caption, labelloc="b").
build_legend(g) -> None — add a subgraph named exactly cluster_legend with
label="Operations". Inside it, for each of the three operations
(improve, combine, repair) create a pair of shape="point" dummy nodes
with unique names prefixed legend_ (e.g. legend_improve_a,
legend_improve_b) and one g_legend.edge(a, b, color=EDGE_COLORS[op], label=op). These dummy nodes must NOT share names with real try nodes and must
NOT connect to them. Placement is best-effort; don't force-pin.
build_caption(title: str = "SensorFlow lineage") -> str — return title
unchanged (the graph caption is just the run title; no shade note).
resolve_output_target(out_path: str) -> tuple[str,str] — split into
(stem_without_ext, graphviz_format); .png→"png", .svg→"svg";
missing/unknown ext → ValueError.


Don't:


Don't set node/edge attrs that belong to CHUNK 6 (fill, border, label).
Don't connect legend dummy nodes to real nodes.


Verify (no dot needed — read g.source): style_edge adds a parent→child
edge with the right color; apply_graph_attrs puts rankdir=TB + label +
labelloc=b in the source; build_legend adds a cluster_legend with three
colored entries; resolve_output_target("x/lineage.png")==("x/lineage","png"),
.svg works, no-ext → ValueError.

Feasibility first: confirm EDGE_COLORS/UNKNOWN_EDGE_COLOR exist from CHUNK 1.


CHUNK 6 — Node styler (integrator) ⚠ Sonnet

Model: Sonnet (subtle graphviz style composition + two documented traps).
Depends on: CHUNKS 1, 2, 3.

Goal: apply ALL of a node's visual channels in one place.

Do: implement


style_node(g, record: TryRecord, frontier: set[str], id_min: int, id_max: int) -> None.
Compute on_frontier = is_frontier(record.try_id, frontier),
role = determine_role(record.status, on_frontier),
fill = temporal_fill_color(record.try_id, id_min, id_max),
border = border_attrs_for_role(role),
label = format_node_label(record.try_id, on_frontier, record.f1_score_class1, record.num_leaf_nodes),
tooltip = build_node_tooltip(record). Then call g.node(record.try_id, label=label, tooltip=tooltip, shape="box", style=<composed>, fillcolor=fill, color=border["color"], penwidth=border["penwidth"]).


Two traps you MUST get right:


Compose style as "filled" plus ,dashed only for error →
"filled,dashed". Setting style="dashed" alone WIPES the temporal fill.
Set shape="box" per node in this call, NOT as a graph node-default —
a node-default only applies to nodes added afterward (ordering trap).


Don't:


Don't move any of the borrowed logic (role/label/tooltip/fill) into this
function — call the CHUNK 1–3 functions.
Don't add operation coloring to the node (that's edges).


Verify (read g.source): frontier node → penwidth=3.0, color=#222222;
dominated → penwidth=1.0, color=#999999; error node → style contains BOTH
filled and dashed; every node has a fillcolor hex, shape=box, a label,
and a tooltip.

Feasibility first: confirm temporal_fill_color, determine_role,
border_attrs_for_role, format_node_label, build_node_tooltip, is_frontier
all exist with the CHUNK 1–3 signatures.


CHUNK 7 — Graph builder

Model: Haiku.  Depends on: CHUNKS 5, 6.

Goal: turn records into a fully-styled graphviz.Digraph, deterministically.

Do: implement


build_graph(records: list[TryRecord], frontier: set[str], id_min: int, id_max: int) -> "graphviz.Digraph".
Create a graphviz.Digraph. Iterate records sorted by int(try_id). For
each record: style_node(...); then for each parent_id in
record.parent_ids: style_edge(g, record.try_id, parent_id, record.operation).
Return g.


Don't:


Don't apply graph attrs or the legend here (CHUNK 5 functions, called later by
the orchestrator).
Don't special-case combine — parent_ids already carries one or two ids; just
loop them (two ids → two edges, same color).


Verify (read g.source): 26 nodes for the real run; combine try 5 has two
incoming edges (from 0 and 3); nodes appear in ascending int(try_id) order;
root has no incoming edge.

Feasibility first: confirm style_node (CHUNK 6) and style_edge (CHUNK 5)
exist with the stated signatures.


CHUNK 8 — Emitter (first dot-binary dependency)

Model: Haiku.  Depends on: CHUNK 5 (resolve_output_target).

Goal: write the image file and return its real path.

Do: implement


emit_graph(g, out_path: str | Path) -> str — resolve_output_target(str(out_path))
→ (stem, fmt); ensure the parent dir exists
(Path(stem).parent.mkdir(parents=True, exist_ok=True)); return
g.render(filename=stem, format=fmt, cleanup=True) directly (its return value
is the written path — no separate path reconstruction).


Don't:


Don't leave the intermediate DOT source file behind (cleanup=True).
Don't swallow graphviz.backend.ExecutableNotFound — let it propagate (it's
the "dot not on PATH" signal).


Verify (needs dot): render a 2-node graph to a temp .png → file exists,
non-empty; .svg path → svg exists; parent dir created if missing; returned path
matches what was written; missing extension → ValueError.

Feasibility first: confirm dot -V works in this environment and
resolve_output_target exists from CHUNK 5.


CHUNK 9 — Orchestrator

Model: Haiku.  Depends on: CHUNKS 4, 5, 6, 7, 8.

Goal: the single public entry point, end to end.

Do: implement


render_lineage(tries_csv_path: str | Path, frontier_try_ids: Iterable[int|str], out_path: str | Path) -> str.
Sequence: records = load_tries(...); if empty → raise
ValueError("no tries to render"); id_min, id_max = compute_id_range(r.try_id for r in records);
frontier = normalize_frontier(frontier_try_ids);
g = build_graph(records, frontier, id_min, id_max);
apply_graph_attrs(g, build_caption()); build_legend(g);
return emit_graph(g, out_path).


Don't:


Don't add parameters (format is inferred from out_path's extension — no
fmt arg).
Don't log to W&B.


Verify (end to end): call directly on the real tries.csv with the real
bare-int frontier list → lineage.png written, non-empty; the known frontier
try_ids show penwidth=3.0 in the pre-emit g.source; empty frontier → renders
with no thick borders; all-error sample → renders, all dashed.

Feasibility first: confirm load_tries, compute_id_range,
normalize_frontier, build_graph, apply_graph_attrs, build_caption,
build_legend, emit_graph all exist with their signatures.


CHUNK 10 — Wire into get_pareto.py ⚠ Sonnet (touches existing code)

Model: Sonnet (edits existing code; must not change plot_pareto's
behavior or return value).  Depends on: CHUNK 9.

Goal: render the lineage graph automatically at run-end, beside pareto.png.

Do:


In get_pareto.py, inside the plot_pareto(...) function, immediately after
the line that writes pareto.png (the fig.savefig(out_path / "pareto.png", ...) call), add a call:
render_lineage(csv_path, frontier_ids, out_path / "lineage.png")
using the csv_path, frontier_ids, and out_path locals already in scope
there. Add the import for render_lineage.
(Optional, only if asked) a second call with out_path / "lineage.svg".


Don't:


Don't change plot_pareto's signature or its return (fig, ax).
Don't modify pareto_frontier_try_ids or any frontier computation.
Don't move or alter the pareto.png savefig itself.
Don't add W&B logging.


Verify: run plot_pareto on a real run dir → BOTH pareto.png and
lineage.png appear in out_path; plot_pareto still returns (fig, ax)
unchanged; nothing else in the run behaves differently.

Feasibility first (scout the real file): confirm in get_pareto.py that
plot_pareto writes pareto.png via a fig.savefig(out_path / "pareto.png", …)
call, and that csv_path, frontier_ids, and out_path are all in scope at
that point. If the variable names or the savefig call differ from what's stated,
STOP and report before editing.
