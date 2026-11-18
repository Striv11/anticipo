//! Exercises the C-ABI surface natively (the same code the browser calls).

use anticipo::session::{symbol_of, Session, VOCAB};
use anticipo::wasm::*;

fn probs() -> Vec<f32> {
    let p = anticipo_predict();
    assert!(!p.is_null());
    unsafe { std::slice::from_raw_parts(p, anticipo_vocab() as usize) }.to_vec()
}

#[test]
fn non_ascii_maps_to_question_mark() {
    assert_eq!(symbol_of(b'a'), 97);
    assert_eq!(symbol_of(200), b'?' as usize);
}

#[test]
fn session_learns_a_repeating_phrase() {
    let mut s = Session::new(3, 24, 0.1, 1);
    let text = b"abcabcabcabc".repeat(30);
    let sur = s.observe_all(&text);
    let early: f32 = sur[..24].iter().sum::<f32>() / 24.0;
    let late: f32 = sur[sur.len() - 24..].iter().sum::<f32>() / 24.0;
    assert!(late < early * 0.25, "early {early} late {late}");
    let p = s.predict();
    assert_eq!(p.len(), VOCAB);
}

#[test]
fn ffi_flow_observe_predict_reset() {
    // Thread-local state: this test owns its thread's session.
    anticipo_init(3, 24, 0.1, 7);
    let p0 = probs();
    assert_eq!(p0.len(), 128);
    assert!((p0.iter().sum::<f32>() - 1.0).abs() < 1e-4);

    let text = b"abcabcabc".repeat(40);
    let input = anticipo_input_buffer(text.len() as u32);
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), input, text.len()) };
    let mean = anticipo_observe_buffer(text.len() as u32);
    assert!(mean >= 0.0 && mean.is_finite());

    // After "...abc" the model should expect 'a'.
    let p = probs();
    let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
    assert_eq!(best, b'a' as usize);

    let s = anticipo_observe(b'a' as u32);
    assert!(s < 0.5, "learned symbol should be unsurprising, got {s}");

    anticipo_reset();
    let fresh = probs();
    assert!(fresh.iter().all(|&x| (x - fresh[0]).abs() < 0.05));
    // Invalid lengths are rejected rather than reading out of bounds.
    anticipo_input_buffer(4);
    assert_eq!(anticipo_observe_buffer(5), -1.0);
    assert_eq!(anticipo_observe_buffer(0), -1.0);
}

#[test]
fn uninitialised_calls_are_safe() {
    // Fresh thread => no session.
    std::thread::spawn(|| {
        assert_eq!(anticipo_observe(65), -1.0);
        assert!(anticipo_predict().is_null());
        anticipo_reset();
        anticipo_reset_context();
    })
    .join()
    .unwrap();
}
