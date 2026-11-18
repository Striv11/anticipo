#!/usr/bin/env bash
# Single quality gate used by CI and by build-lab before every push:
# format check, lint, tests.
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
# WASM build (skipped locally if the target is not installed; CI installs it).
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  cargo build --release --target wasm32-unknown-unknown --lib
else
  echo "note: wasm32-unknown-unknown not installed, skipping wasm build" >&2
fi
