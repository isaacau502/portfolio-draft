Sprint 1 (wk 1–2) — Algorithm core + proof. Goal: the search is good, and provably so.

Cleanup/stabilize (finish) — d1
Lineage graph + W&B port — d1–2
Probability tuning (off lineage) — d2 (half)
Cascade evaluation — d3–4
Artifact side-channel — d4 (half)
Budget feature — d5–6
Model selector — d7
LangSmith eval integration (tracing + datasets + scorers + CI gate) — d8–10
Slack/overrun: baked into d10

End state: improved, cost-aware, eval-proven multi-objective search. This is already a complete, strong artifact — your safety net if later sprints slip.
Sprint 2 (wk 3–4) — Orchestration. Goal: scalable, durable search. Highest risk sprint.

Prompt + instruction tuning (now measurable via evals) — d1
LangGraph migration (loop → stateful graph; durability + parallel fan-out) — d2–7 (the blow-up risk; writeups say people rebuild 2–3×)
Islands + MAP-Elites (parallel populations, quality-diversity; built with the migration) — d8–10

End state: orchestrated, parallel, durable search. If anything slips, it slips here — LangGraph is where pace dies.
Sprint 3 (wk 5–6) — Productionize + host. Goal: deployable tool, not a script.

Robustness paydown (the deferred debt: real error handling on the paths that currently just raise) — d1–2
Test suite (unit on the pure functions, integration on the loop) — d3–4
Packaging + CLI polish + docs/README — d5
AgentCore deployment (serverless hosting of the LangGraph app) — d6–9 (wide error bars)
Buffer / demo prep / resume writeup — d10
