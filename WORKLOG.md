# Work log

Newest entries at the bottom. Each entry records what actually changed, what was learned, and what comes next.

## 2026-11-16 (day 1)
- Chose anticipo (Rust -> WASM predictive-coding sequence learner) from the Cognition & AI backlog; recent projects were TypeScript, so a Rust core rotates the language.
- Implemented the two-layer predictive-coding network with local error-driven inference and learning, a deterministic RNG, and integration tests (melody learning, falling surprise, context-length sensitivity, determinism, input validation).
- Learned: my first "ambiguous" test sequence was itself ambiguous at order 2 (caught by the test); fixed the sequence rather than the model.
- CI runs `check.sh` with fmt, clippy and tests; Pages workflow serves a placeholder page.
- Next (day 2): expose per-layer errors, add tokenizers and a Markov baseline to benchmark against.

## 2026-11-17 (day 2)
- `Step` now exposes `output_error` and `hidden_error` (L2 norms of the layer prediction errors); test checks they shrink as a melody is learned.
- Added `CharTokenizer`, `MelodyTokenizer` (property-style round-trip tests with the seeded RNG) and an online `MarkovBaseline`.
- Added `tests/benchmark.rs`: PC model vs order-1 Markov baseline on a noisy source; passes within 10% (1.13 vs 1.11 bits/symbol) and prints bits/symbol.
- Learned: the benchmark exposed a real flaw. Output weights learned from the *settled* error, which relaxation shrinks, so on noisy data the model grew overconfident and loss got worse with more data (1.2 -> 1.3 bits). Learning from the prediction error fixed that but slowed learning on deterministic data enough to break an existing test; averaging the settled and top-down presynaptic activity satisfied both without touching the tests.
- Caveat: at 30000 symbols the gap to the baseline is ~7%, still inside the budget but drifting. Worth a decay or lr schedule later.
- Next (day 3): WASM bindings, CI wasm32 build, minimal UI.
