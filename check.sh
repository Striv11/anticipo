#!/usr/bin/env bash
# Single quality gate used by CI and by build-lab before every push:
# format check, lint, tests.
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
# Latency budget (<2 ms/observe at hidden=64) is only meaningful on an optimised build.
cargo test --release --test perf_budget
# WASM build (skipped locally if the target is not installed; CI installs it).
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  cargo build --release --target wasm32-unknown-unknown --lib
  size=$(stat -c %s target/wasm32-unknown-unknown/release/anticipo.wasm)
  echo "wasm size: $size bytes (budget 307200)"
  if [ "$size" -ge 307200 ]; then echo "error: WASM exceeds 300 KB budget" >&2; exit 1; fi
else
  echo "note: wasm32-unknown-unknown not installed, skipping wasm build" >&2
fi
