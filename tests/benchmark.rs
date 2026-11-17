//! Online-vs-baseline benchmark on a first-order Markov source.
use anticipo::rng::Rng;
use anticipo::{Config, MarkovBaseline, Model};

const VOCAB: usize = 4;

/// Sparse-ish transition matrix with a dominant successor per state.
fn source(n: usize, seed: u64) -> Vec<usize> {
    let probs: [[f32; VOCAB]; VOCAB] = [
        [0.05, 0.80, 0.10, 0.05],
        [0.10, 0.05, 0.80, 0.05],
        [0.05, 0.10, 0.05, 0.80],
        [0.70, 0.10, 0.10, 0.10],
    ];
    let mut rng = Rng::new(seed);
    let mut state = 0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.uniform();
        let mut acc = 0.0;
        let mut next = VOCAB - 1;
        for (j, p) in probs[state].iter().enumerate() {
            acc += p;
            if u < acc {
                next = j;
                break;
            }
        }
        out.push(next);
        state = next;
    }
    out
}

fn bits(nats: f32) -> f32 {
    nats / std::f32::consts::LN_2
}

#[test]
fn pc_model_matches_markov_baseline_within_ten_percent() {
    let data = source(6000, 42);
    let tail = 2000;
    let mut cfg = Config::new(VOCAB, 1);
    cfg.learning_rate = 0.005;
    let mut pc = Model::new(cfg);
    let mut mk = MarkovBaseline::new(VOCAB, 1, 0.5);
    let (mut pc_loss, mut mk_loss) = (0.0, 0.0);
    for (i, &s) in data.iter().enumerate() {
        let a = pc.observe(s).surprise;
        let b = mk.observe(s);
        if i >= data.len() - tail {
            pc_loss += a;
            mk_loss += b;
        }
    }
    let (pc_bps, mk_bps) = (bits(pc_loss / tail as f32), bits(mk_loss / tail as f32));
    println!("bits/symbol  pc={pc_bps:.4}  markov={mk_bps:.4}");
    assert!(
        pc_bps <= mk_bps * 1.10,
        "pc {pc_bps} vs markov {mk_bps} bits/symbol"
    );
}
