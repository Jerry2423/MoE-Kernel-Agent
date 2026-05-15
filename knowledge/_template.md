---
topic: <short-slug>
applies_to: <shape thresholds, workload characteristics, or "general">
source: <comma-separated paths in context/ or .claude/skills/>
confidence: <high | medium | low>
---

# <Human-readable title>

## Summary

<At most 3 sentences. What is this technique and when does it matter.>

## When to use

<Concrete triggers: shape ranges, bottleneck class, workload properties. Avoid "always" / "often" — give thresholds.>

## Code pattern

<Minimal Triton snippet or pseudocode demonstrating the core idea. Do NOT paste a full kernel. Keep under ~30 lines.>

```python
# example
```

## Tradeoffs

<What this costs: register pressure, shared memory, compile time, correctness risk, numerical stability, code complexity.>

## Pitfalls

<Concrete failure modes observed in the source material. Cite line numbers or subsections where possible.>
