use anticipo::{Config, Model};

fn run(model: &mut Model, seq: &[usize], reps: usize) -> (f32, f32) {
    // Returns (accuracy, mean surprise) over the final repetition.
    let mut hits = 0;
    let mut total = 0.0;
    for r in 0..reps {
        for &s in seq {
            let step = model.observe(s);
            if r == reps - 1 {
                hits += step.correct as usize;
                total += step.surprise;
            }
        }
    }
    (hits as f32 / seq.len() as f32, total / seq.len() as f32)
}

#[test]
fn learns_a_repeating_melody() {
    let seq = [0, 1, 2, 3, 0, 1, 2, 3];
    let mut m = Model::new(Config::new(4, 1));
    let (acc, surprise) = run(&mut m, &seq, 60);
    assert!(acc >= 0.99, "accuracy {acc}");
    assert!(surprise < 0.2, "surprise {surprise}");
}

#[test]
fn surprise_falls_with_experience() {
    let seq = [0, 1, 2, 1, 0, 3, 2, 3];
    let mut m = Model::new(Config::new(4, 2));
    let first: f32 = seq.iter().map(|&s| m.observe(s).surprise).sum();
    let (_, _) = run(&mut m, &seq, 100);
    let last: f32 = seq.iter().map(|&s| m.observe(s).surprise).sum();
    assert!(last < first * 0.25, "first {first} last {last}");
}

#[test]
fn needs_longer_context_when_ambiguous() {
    // Each symbol has two successors; only the previous symbol disambiguates.
    let seq = [0, 1, 2, 3, 1, 0, 3];
    let mut short = Model::new(Config::new(4, 1));
    let mut long = Model::new(Config::new(4, 2));
    let (a_short, _) = run(&mut short, &seq, 200);
    let (a_long, _) = run(&mut long, &seq, 200);
    assert!(a_long >= 0.99, "window 2 accuracy {a_long}");
    assert!(a_short < a_long, "window 1 {a_short} vs {a_long}");
}

#[test]
fn predictions_are_distributions_and_deterministic() {
    let mut a = Model::new(Config::new(5, 2));
    let mut b = Model::new(Config::new(5, 2));
    for s in [0, 3, 1, 4, 2, 2, 0] {
        let (sa, sb) = (a.observe(s), b.observe(s));
        assert_eq!(sa.prediction, sb.prediction);
        let sum: f32 = sa.prediction.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(sa.surprise.is_finite() && sa.surprise >= 0.0);
    }
}

#[test]
#[should_panic(expected = "symbol out of range")]
fn rejects_out_of_range_symbol() {
    Model::new(Config::new(3, 1)).observe(3);
}
