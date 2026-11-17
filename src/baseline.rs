//! Order-k Markov baseline with additive smoothing, learned online.
//!
//! Used as a reference: a predictive-coding model on a Markov source should
//! approach this model's log-loss.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct MarkovBaseline {
    vocab: usize,
    order: usize,
    alpha: f32,
    counts: HashMap<Vec<usize>, Vec<f32>>,
    history: Vec<usize>,
}

impl MarkovBaseline {
    pub fn new(vocab: usize, order: usize, alpha: f32) -> Self {
        assert!(vocab >= 2 && alpha > 0.0);
        MarkovBaseline {
            vocab,
            order,
            alpha,
            counts: HashMap::new(),
            history: vec![0; order],
        }
    }

    pub fn predict(&self) -> Vec<f32> {
        let zero = vec![0.0; self.vocab];
        let c = self.counts.get(&self.history).unwrap_or(&zero);
        let total: f32 = c.iter().sum::<f32>() + self.alpha * self.vocab as f32;
        c.iter().map(|&n| (n + self.alpha) / total).collect()
    }

    /// Returns the surprise (nats) of `symbol`, then learns it.
    pub fn observe(&mut self, symbol: usize) -> f32 {
        assert!(symbol < self.vocab, "symbol out of range");
        let surprise = -self.predict()[symbol].ln();
        self.counts
            .entry(self.history.clone())
            .or_insert_with(|| vec![0.0; self.vocab])[symbol] += 1.0;
        if self.order > 0 {
            self.history.remove(0);
            self.history.push(symbol);
        }
        surprise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictions_form_a_distribution() {
        let mut m = MarkovBaseline::new(5, 2, 0.5);
        for s in [0, 1, 2, 3, 4, 0, 1] {
            m.observe(s);
            let p = m.predict();
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn learns_deterministic_cycle() {
        let mut m = MarkovBaseline::new(3, 1, 0.1);
        let mut last = 0.0;
        for i in 0..90 {
            last = m.observe(i % 3);
        }
        assert!(last < 0.05, "{last}");
    }

    #[test]
    fn order_zero_tracks_frequencies() {
        let mut m = MarkovBaseline::new(2, 0, 1.0);
        for _ in 0..30 {
            m.observe(1);
        }
        assert!(m.predict()[1] > 0.9);
    }
}
