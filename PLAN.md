# Plan

Seven-day arc. Every milestone has acceptance criteria that can be checked by a test or a command.

| Day | Milestone | Acceptance criteria | Status |
|-----|-----------|---------------------|--------|
| 1 | Core model: 2-layer predictive-coding net in Rust, deterministic RNG, CI | `bash check.sh` passes; tests show a repeating melody is learned (>=99% top-1), surprise falls >75% with experience, a window-2 model beats window-1 on an ambiguous sequence, predictions sum to 1 and are deterministic per seed | [x] |
| 2 | Richer model + data: per-layer error exposure, text/melody tokenizers, online-vs-baseline benchmark | `Step` exposes layer error magnitudes; tokenizers round-trip (property tests); an order-k Markov baseline is implemented and the PC model matches it on a Markov source within 10% of its log-loss; `cargo bench`-style test prints bits/symbol | [ ] |
| 3 | WASM bindings + minimal UI | `wasm32-unknown-unknown` build in CI; JS API `observe/predict/reset` with typed arrays; page loads the bundle and shows next-symbol probability bars for typed text; Pages deploys | [ ] |
| 4 | Melody mode + audio | Piano-roll input (mouse/touch/keys), Web Audio playback of the model's predicted continuation; "surprise" trace chart updates live; keyboard shortcuts documented in-app | [ ] |
| 5 | Visualizing the hierarchy | Live view of hidden activity and error units per layer; learning-rate/window/hidden-size controls rebuild the model; presets (scale, nursery rhyme, English text) | [ ] |
| 6 | Product polish | Responsive layout verified at 375px and 1280px, light/dark, empty/loading/error states, landing section, save/load model state to localStorage; screenshots in README | [ ] |
| 7 | Tests and budgets | Playwright e2e for core flows (type text, see predictions fall in surprise, melody playback toggles, reset); benchmark with budgets (<2 ms/observe at hidden=64, WASM <300 KB) enforced in CI; final docs | [ ] |

## Stretch
- Third layer with different time scales (slow context layer).
- Export/import the learned weights as a file.
