//! anticipo: an online predictive-coding sequence learner.
//!
//! A small hierarchical generative model predicts the next symbol from the
//! last `window` symbols. Every layer keeps a *prediction error* unit; both
//! inference (settling hidden activity) and learning (weight updates) use only
//! locally available errors, as in Rao & Ballard / Whittington & Bogacz.

pub mod model;
pub mod rng;

pub use model::{Config, Model, Step};
