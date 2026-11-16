# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-16 (day 1)
- Chose anticipo (Rust -> WASM predictive-coding sequence learner) from the Cognition & AI backlog; recent projects were TypeScript, so a Rust core rotates the language.
- Implemented the two-layer predictive-coding network with local error-driven inference and learning, a deterministic RNG, and integration tests (melody learning, falling surprise, context-length sensitivity, determinism, input validation).
- Learned: my first "ambiguous" test sequence was itself ambiguous at order 2 (caught by the test); fixed the sequence rather than the model.
- CI runs `check.sh` with fmt, clippy and tests; Pages workflow serves a placeholder page.
- Next (day 2): expose per-layer errors, add tokenizers and a Markov baseline to benchmark against.
