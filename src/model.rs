//! Two-layer predictive-coding network.
//!
//! Layers (top to bottom): context `x2` (one-hot window, clamped) -> hidden
//! `x1` (free) -> output `x0` (the next symbol, clamped to the truth while
//! learning).
//!
//! ```text
//!   mu1 = W1 x2            e1 = x1 - mu1
//!   mu0 = W0 tanh(x1)      e0 = onehot(next) - softmax(mu0)
//! ```
//!
//! Inference relaxes `x1` by gradient descent on the summed squared error;
//! learning then applies the local rules `dW0 = lr e0 tanh(x1)^T`,
//! `dW1 = lr e1 x2^T`.

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Config {
    /// Alphabet size (number of distinct symbols).
    pub vocab: usize,
    /// How many previous symbols form the context.
    pub window: usize,
    pub hidden: usize,
    /// Inference relaxation steps per observation.
    pub infer_steps: usize,
    /// Inference step size.
    pub infer_rate: f32,
    pub learning_rate: f32,
    pub seed: u64,
}

impl Config {
    pub fn new(vocab: usize, window: usize) -> Self {
        Config {
            vocab,
            window,
            hidden: 24,
            infer_steps: 12,
            infer_rate: 0.2,
            learning_rate: 0.1,
            seed: 1,
        }
    }
}

/// Result of observing one symbol.
#[derive(Clone, Debug)]
pub struct Step {
    /// Probability the model assigned to each symbol *before* seeing it.
    pub prediction: Vec<f32>,
    /// Surprise of the actual symbol in nats: `-ln p(actual)`.
    pub surprise: f32,
    /// Whether the arg-max prediction matched the symbol.
    pub correct: bool,
    /// L2 norm of the output-layer error `onehot(actual) - prediction`
    /// before any relaxation (how wrong the feed-forward guess was).
    pub output_error: f32,
    /// L2 norm of the hidden-layer error `x1 - mu1` after relaxation (how far
    /// the hidden state had to move from its top-down prediction to explain
    /// the observation).
    pub hidden_error: f32,
    /// Settled hidden activity `tanh(x1)`, one value per hidden unit.
    pub hidden_activity: Vec<f32>,
    /// Per-unit hidden prediction errors `x1 - mu1` (the vector whose norm is
    /// `hidden_error`).
    pub hidden_errors: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Model {
    cfg: Config,
    /// hidden x (window*vocab)
    w1: Vec<f32>,
    /// vocab x hidden
    w0: Vec<f32>,
    history: Vec<usize>,
}

const MAGIC: &[u8; 4] = b"ANT1";

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter_mut().for_each(|v| *v /= s);
    e
}

impl Model {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.vocab >= 2, "vocab must be >= 2");
        assert!(cfg.window >= 1 && cfg.hidden >= 1);
        let mut rng = Rng::new(cfg.seed);
        let in_dim = cfg.window * cfg.vocab;
        let w1 = (0..cfg.hidden * in_dim)
            .map(|_| rng.symmetric(0.3))
            .collect();
        let s0 = (1.0 / cfg.hidden as f32).sqrt();
        let w0 = (0..cfg.vocab * cfg.hidden)
            .map(|_| rng.symmetric(s0))
            .collect();
        // Pad history with symbol 0 so the first prediction is well defined.
        let history = vec![0; cfg.window];
        Model {
            cfg,
            w1,
            w0,
            history,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Indices of active inputs in the one-hot window encoding.
    fn active_inputs(&self) -> Vec<usize> {
        let v = self.cfg.vocab;
        self.history
            .iter()
            .enumerate()
            .map(|(slot, &s)| slot * v + s)
            .collect()
    }

    /// Feed-forward top-down prediction of the hidden layer.
    fn hidden_prediction(&self, active: &[usize]) -> Vec<f32> {
        let in_dim = self.cfg.window * self.cfg.vocab;
        (0..self.cfg.hidden)
            .map(|h| active.iter().map(|&i| self.w1[h * in_dim + i]).sum())
            .collect()
    }

    fn output_logits(&self, x1: &[f32]) -> Vec<f32> {
        let hd = self.cfg.hidden;
        (0..self.cfg.vocab)
            .map(|o| (0..hd).map(|h| self.w0[o * hd + h] * x1[h].tanh()).sum())
            .collect()
    }

    /// Probability distribution over the next symbol given current history.
    pub fn predict(&self) -> Vec<f32> {
        let active = self.active_inputs();
        let mu1 = self.hidden_prediction(&active);
        softmax(&self.output_logits(&mu1))
    }

    /// Observe the true next symbol: score the prediction, learn from the
    /// error, then advance the context window.
    pub fn observe(&mut self, symbol: usize) -> Step {
        assert!(symbol < self.cfg.vocab, "symbol out of range");
        let (vocab, hd) = (self.cfg.vocab, self.cfg.hidden);
        let in_dim = self.cfg.window * vocab;
        let active = self.active_inputs();
        let mu1 = self.hidden_prediction(&active);
        let prediction = softmax(&self.output_logits(&mu1));
        let surprise = -prediction[symbol].max(1e-9).ln();
        let correct = prediction
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            == Some(symbol);

        // Inference: relax hidden activity with the output clamped to truth.
        let mut x1 = mu1.clone();
        let mut e0 = vec![0.0; vocab];
        let output_error = (0..vocab)
            .map(|o| {
                let e = if o == symbol { 1.0 } else { 0.0 } - prediction[o];
                e * e
            })
            .sum::<f32>()
            .sqrt();
        for _ in 0..self.cfg.infer_steps {
            let p = softmax(&self.output_logits(&x1));
            for o in 0..vocab {
                e0[o] = if o == symbol { 1.0 } else { 0.0 } - p[o];
            }
            for h in 0..hd {
                let back: f32 = (0..vocab).map(|o| self.w0[o * hd + h] * e0[o]).sum();
                let t = x1[h].tanh();
                let e1 = x1[h] - mu1[h];
                x1[h] += self.cfg.infer_rate * (-e1 + back * (1.0 - t * t));
            }
        }

        // Learning: local, error-driven updates at the settled state.
        // The output weights learn from the error of the *prediction* (made
        // from the top-down state), not the settled error: once x1 has been
        // pulled toward the truth the residual output error shrinks, which on
        // noisy sequences makes the model drift toward overconfidence. The
        // presynaptic activity is the mean of the settled and top-down states,
        // which keeps learning fast on deterministic sequences.
        let lr = self.cfg.learning_rate;
        for o in 0..vocab {
            let e = if o == symbol { 1.0 } else { 0.0 } - prediction[o];
            for h in 0..hd {
                self.w0[o * hd + h] += lr * e * (0.5 * (x1[h].tanh() + mu1[h].tanh()));
            }
        }
        for h in 0..hd {
            let e1 = x1[h] - mu1[h];
            for &i in &active {
                self.w1[h * in_dim + i] += lr * e1;
            }
        }

        let hidden_error = (0..hd)
            .map(|h| (x1[h] - mu1[h]).powi(2))
            .sum::<f32>()
            .sqrt();

        let hidden_activity = x1.iter().map(|v| v.tanh()).collect();
        let hidden_errors = (0..hd).map(|h| x1[h] - mu1[h]).collect();

        self.history.remove(0);
        self.history.push(symbol);
        Step {
            prediction,
            surprise,
            correct,
            output_error,
            hidden_error,
            hidden_activity,
            hidden_errors,
        }
    }

    /// Serialise the learned weights: magic `ANT1`, then `vocab`, `window`,
    /// `hidden` as u32 LE, then `w1` and `w0` as f32 LE.
    pub fn export_weights(&self) -> Vec<u8> {
        let c = &self.cfg;
        let mut out = Vec::with_capacity(16 + 4 * (self.w1.len() + self.w0.len()));
        out.extend_from_slice(MAGIC);
        for d in [c.vocab, c.window, c.hidden] {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for w in self.w1.iter().chain(&self.w0) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Load weights written by [`Model::export_weights`]. The shape must
    /// match this model's configuration; on error the model is unchanged.
    /// The context window is cleared.
    pub fn import_weights(&mut self, bytes: &[u8]) -> Result<(), String> {
        let c = &self.cfg;
        if bytes.len() < 16 || &bytes[..4] != MAGIC {
            return Err("not an anticipo state".into());
        }
        let dim = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        let (v, w, h) = (dim(0) as usize, dim(1) as usize, dim(2) as usize);
        if (v, w, h) != (c.vocab, c.window, c.hidden) {
            return Err(format!(
                "shape mismatch: state is vocab {v}, window {w}, hidden {h}; model is vocab {}, window {}, hidden {}",
                c.vocab, c.window, c.hidden
            ));
        }
        let n1 = self.w1.len();
        if bytes.len() != 16 + 4 * (n1 + self.w0.len()) {
            return Err("state has the wrong length".into());
        }
        let vals: Vec<f32> = bytes[16..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if vals.iter().any(|v| !v.is_finite()) {
            return Err("state contains non-finite weights".into());
        }
        self.w1.copy_from_slice(&vals[..n1]);
        self.w0.copy_from_slice(&vals[n1..]);
        self.reset_context();
        Ok(())
    }

    /// Clear the context window (weights are kept).
    pub fn reset_context(&mut self) {
        self.history = vec![0; self.cfg.window];
    }
}
