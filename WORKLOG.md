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

## 2026-11-18 (day 3)
- Added `Session` (ASCII vocab of 128) and a C-ABI in `src/wasm.rs`: init, observe (single char or UTF-8 buffer), predict (pointer to an f32 array), reset. Deliberately no wasm-bindgen: keeps the core dependency-free and the binary ~40 KB.
- `web/anticipo.js` wraps the exports with typed-array views; `web/index.html` shows top-8 next-character bars and last-input surprise as you type, with loading/error states and a reset button.
- `scripts/build-web.sh` builds into `web/pkg` (gitignored); the Pages workflow now runs it; `check.sh` also builds for wasm32 when the target is installed.
- Tests (`tests/wasm_api.rs`) call the FFI natively: learning a phrase, distribution validity, reset, bounds-checked buffer, uninitialised calls. Also smoke-tested the real .wasm in Node (p('a') = 0.99 after `abc` x40).
- Learned: exports are plain `extern "C"` functions, so clippy and tests cover them natively without a wasm runner. Not verified: the page in a real browser (no browser here); Pages deployment only happens on push.
- Next (day 4): melody mode, piano roll, Web Audio playback, surprise trace chart.
