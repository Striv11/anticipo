#!/usr/bin/env bash
# Build the WASM core and copy it next to the web UI (web/pkg/anticipo.wasm).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown --lib
mkdir -p web/pkg
cp target/wasm32-unknown-unknown/release/anticipo.wasm web/pkg/anticipo.wasm
ls -l web/pkg/anticipo.wasm
