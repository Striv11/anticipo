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
    vocab: () => x.anticipo_vocab(),
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
    resetContext: () => x.anticipo_reset_context(),
    reset: () => x.anticipo_reset(),
  };
}
