# Agent trajectory — 2026-04-18 DeepSeek-V3 MoE on B200

This directory captures the full trajectory of one agent run. Everything produced and consumed during that run is contained here:

- `context/` — Phase-1 input material the agent was given
- `knowledge/` — Phase-1 output knowledge base the agent produced
- `HANDOFF.md` — phase-by-phase handoff log
- `ITERATIONS.md` — Phase-3 iteration log
- `trajectory/` — per-iteration captured artifacts
- `solution/` — final solution + design docs

The submitted kernel lives at `solution/triton/kernel.py`.
