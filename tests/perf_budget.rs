//! Performance budget: observe() latency at hidden=64 on the text vocabulary (128 symbols).
//! The 2 ms budget applies to optimised builds (what ships to WASM); unoptimised test builds
//! get a 25x allowance so the test still catches order-of-magnitude regressions.
use anticipo::{Config, Model};
use std::time::Instant;

#[test]
fn observe_latency_within_budget_at_hidden_64() {
    let mut cfg = Config::new(128, 4);
    cfg.hidden = 64;
    let mut m = Model::new(cfg);
    let text = b"the quick brown fox jumps over the lazy dog. ";
    for i in 0..200 {
        m.observe(text[i % text.len()] as usize); // warm-up
    }
    let n = 1000;
    let t = Instant::now();
    for i in 0..n {
        m.observe(text[i % text.len()] as usize);
    }
    let per_ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let budget = if cfg!(debug_assertions) { 50.0 } else { 2.0 };
    println!("observe: {per_ms:.4} ms (budget {budget} ms)");
    assert!(
        per_ms < budget,
        "observe took {per_ms:.4} ms, budget {budget} ms"
    );
}
