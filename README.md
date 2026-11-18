# anticipo

> A brain-inspired sequence learner that predicts the next note or letter, learns online from its own prediction errors, and shows you how surprised it is. Rust compiled to WASM.

Live demo: _planned_ (https://striv11.github.io/anticipo/ once the UI ships on day 3).

![CI](https://github.com/Striv11/anticipo/actions/workflows/ci.yml/badge.svg)

## Problem
Most sequence models are trained offline on huge corpora with backpropagation through time, and you only see their final answer. It is hard to build intuition for *how* a brain-like learner discovers structure in a melody or a text stream as it goes. anticipo is a small, inspectable model you can feed a sequence and watch learn: its next-symbol predictions, its surprise, and its per-layer prediction errors.

## Motivation
Predictive coding says cortex continually predicts its input and learns from the mismatch (the prediction error). Its inference and weight updates are *local*: each unit only needs its own error and its neighbours' activity, with no global backward pass. Building this from scratch in Rust explores:
- local error-driven learning and iterative inference (relaxation of hidden state),
- online learning, one observation at a time, with no dataset or batches,
- shipping the same deterministic Rust core to the browser via WASM, with no network and no data leaving the device.

## Architecture
```
 context window (one-hot, last K symbols)   x2  (clamped)
        | W1  top-down prediction mu1
        v
 hidden state x1  <-- relaxed to reduce errors e1 = x1 - mu1 and e0
        | W0  prediction mu0 = W0 tanh(x1) -> softmax
        v
 next symbol x0   (e0 = onehot(actual) - softmax(mu0))
```
- `src/model.rs`: the network. `predict()` is a pure feed-forward guess; `observe(sym)` scores that guess (surprise = -ln p), relaxes the hidden layer with the truth clamped, applies local updates `dW0 = lr*e0*tanh(x1)^T`, `dW1 = lr*e1*x2^T`, and slides the window.
- `src/rng.rs`: dependency-free deterministic xorshift RNG; the same seed gives identical runs.
- `src/tokenizer.rs`: `CharTokenizer` (text) and `MelodyTokenizer` (note names like `F#3`), both round-trip tested.
- `src/baseline.rs`: online order-k Markov model with additive smoothing, the reference for benchmarks. `Step` also reports `output_error` and `hidden_error` norms per observation.
- Benchmark (`cargo test --test benchmark -- --nocapture`): on a first-order Markov source the PC model reaches ~1.13 bits/symbol vs 1.11 for the baseline (6000 symbols); at 30000 symbols the gap widens to ~7%, since constant-rate learning keeps fluctuating.
- `src/session.rs` + `src/wasm.rs`: ASCII text session and a dependency-free C-ABI (`anticipo_init/observe/predict/reset...`) exported to WASM; `web/anticipo.js` wraps it with typed arrays (no wasm-bindgen). The WASM binary is ~40 KB.
- Invariants: predictions always form a probability distribution; output is deterministic per seed and input.

## Setup
Prerequisites: Rust stable (with `rustfmt` and `clippy`).
```
bash check.sh        # fmt check, clippy -D warnings, tests
cargo test           # tests only
```

## Demo
Build and serve locally:
```
bash scripts/build-web.sh     # writes web/pkg/anticipo.wasm
python3 -m http.server -d web 8000   # open http://localhost:8000
```
Type text in the box; bars show the model's top-8 next-character probabilities and the surprise of your last input. Try typing `abcabcabc...` and watch `a` dominate. The Pages workflow builds the same bundle. Melody mode comes on day 4. The longer target two-minute demo: open the page, pick the "nursery rhyme" preset, watch surprise drop to near zero after a few repeats, then type a deviation and see the spike.

## Design decisions & trade-offs
- Dependency-free core: small WASM size and fully reproducible runs, at the cost of hand-written linear algebra (fine at these sizes).
- Softmax output with cross-entropy-style error instead of Gaussian output error: better-behaved for categorical symbols.
- Fixed number of relaxation steps (12) per observation: predictable latency over exact convergence.

## Roadmap
- [x] Day 1: core predictive-coding model, tests, CI
- [x] Day 2: layer-error exposure, tokenizers, Markov baseline
- [x] Day 3: WASM bindings and minimal web UI
- [ ] Day 4: melody mode with audio
- [ ] Day 5: hierarchy visualization and controls
- [ ] Day 6: polish, responsive design, screenshots
- [ ] Day 7: Playwright e2e and performance budgets

See [PLAN.md](PLAN.md).

## License
MIT. See [LICENSE](LICENSE).
