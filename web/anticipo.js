// Thin JS wrapper over the C-ABI exports in src/wasm.rs.
export async function load(url = "pkg/anticipo.wasm") {
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), {});
  const x = instance.exports;
  const enc = new TextEncoder();
  // Memory may grow, so always create views from the current buffer.
  return {
    init(opts = {}) {
      const { window = 3, hidden = 24, learningRate = 0.1, seed = 1 } = opts;
      x.anticipo_init(window, hidden, learningRate, seed);
    },
    /** Melody mode: symbols are pitch ids (C0 = 0, MIDI = id + 12). */
    initMelody(opts = {}) {
      const { window = 3, hidden = 32, learningRate = 0.1, seed = 1 } = opts;
      x.anticipo_init_melody(window, hidden, learningRate, seed);
    },
    /** Learn one symbol id; returns surprise in nats (-1 if invalid). */
    observeSymbol: (id) => x.anticipo_observe_symbol(id),
    /** The model's imagined next `n` symbol ids (model unchanged). */
    continuation(n) {
      const p = x.anticipo_continue(n);
      if (!p) return [];
      return Array.from(new Uint32Array(x.memory.buffer, p, n));
    },
    vocab: () => x.anticipo_vocab(),
    /** Hidden state after the last observation: {activity, errors} (copies). */
    layers() {
      const n = x.anticipo_hidden();
      const p = x.anticipo_layers();
      if (!p) return { activity: new Float32Array(0), errors: new Float32Array(0) };
      const all = new Float32Array(x.memory.buffer, p, 2 * n);
      return { activity: all.slice(0, n), errors: all.slice(n) };
    },
    /** Next-symbol probabilities as a copied Float32Array. */
    predict() {
      const p = x.anticipo_predict();
      return new Float32Array(x.memory.buffer, p, x.anticipo_vocab()).slice();
    },
    /** Learn from a string; returns mean surprise in nats. */
    observe(text) {
      const bytes = enc.encode(text);
      if (bytes.length === 0) return 0;
      const ptr = x.anticipo_input_buffer(bytes.length);
      new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
      return x.anticipo_observe_buffer(bytes.length);
    },
    /** Learn one character; returns its surprise in nats. */
    observeChar: (c) => x.anticipo_observe(c.charCodeAt(0)),
    /** Learned weights as bytes (a copy), for saving. */
    saveState() {
      const n = x.anticipo_state_export();
      const p = x.anticipo_state_ptr();
      return new Uint8Array(x.memory.buffer, p, n).slice();
    },
    /** Restore weights from saveState(); the model shape must match. Returns success. */
    loadState(bytes) {
      const p = x.anticipo_state_buffer(bytes.length);
      if (!p) return false;
      new Uint8Array(x.memory.buffer, p, bytes.length).set(bytes);
      return x.anticipo_state_import() === 1;
    },
    resetContext: () => x.anticipo_reset_context(),
    reset: () => x.anticipo_reset(),
  };
}
