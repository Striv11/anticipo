//! A text-mode session: an ASCII [`Model`] plus the bookkeeping the web UI
//! needs. Kept free of FFI so it can be tested natively; `wasm.rs` is a thin
//! C-ABI shell around it.

use crate::model::{Config, Model, Step};

/// ASCII vocabulary size. Non-ASCII input is mapped to `?`.
pub const VOCAB: usize = 128;

/// Map a byte to a symbol index (`?` for anything outside ASCII).
pub fn symbol_of(byte: u8) -> usize {
    if byte < 128 {
        byte as usize
    } else {
        b'?' as usize
    }
}

pub struct Session {
    model: Model,
}

impl Session {
    pub fn new(window: usize, hidden: usize, learning_rate: f32, seed: u64) -> Self {
        let mut cfg = Config::new(VOCAB, window.max(1));
        cfg.hidden = hidden.max(1);
        cfg.learning_rate = learning_rate;
        cfg.seed = seed;
        Session {
            model: Model::new(cfg),
        }
    }

    /// Next-symbol distribution over [`VOCAB`] symbols.
    pub fn predict(&self) -> Vec<f32> {
        self.model.predict()
    }

    /// Learn from one byte of input.
    pub fn observe(&mut self, byte: u8) -> Step {
        self.model.observe(symbol_of(byte))
    }

    /// Feed a whole string; returns the surprise (nats) of each byte.
    pub fn observe_all(&mut self, bytes: &[u8]) -> Vec<f32> {
        bytes.iter().map(|&b| self.observe(b).surprise).collect()
    }

    pub fn reset_context(&mut self) {
        self.model.reset_context();
    }

    /// Forget everything: fresh weights from the same configuration.
    pub fn reset(&mut self) {
        self.model = Model::new(self.model.config().clone());
    }
}
