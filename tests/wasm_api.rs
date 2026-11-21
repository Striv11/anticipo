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

#[test]
fn melody_session_learns_and_imagines_continuation() {
    use anticipo::MelodyTokenizer;
    let tune = MelodyTokenizer::encode("C4 E4 G4 E4 D4 F4 A4 F4").unwrap();
    let mut s = Session::melody(2, 24, 0.1, 3);
    assert_eq!(s.vocab(), MelodyTokenizer::VOCAB);
    for _ in 0..60 {
        for &n in &tune {
            s.observe_symbol(n).unwrap();
        }
    }
    assert!(s.observe_symbol(MelodyTokenizer::VOCAB).is_none());
    // After a full pass the context is the end of the tune; the model should
    // imagine the tune starting over.
    let before = s.predict();
    let cont = s.continuation(8);
    assert_eq!(cont, tune);
    // Imagining must not change the real model.
    assert_eq!(before, s.predict());
}

#[test]
fn ffi_melody_mode() {
    anticipo_init_melody(2, 24, 0.1, 5);
    assert_eq!(anticipo_vocab(), 108);
    assert!(probs().len() == 108);
    assert!(anticipo_observe_symbol(60 - 12) >= 0.0);
    assert_eq!(anticipo_observe_symbol(108), -1.0);
    assert!(anticipo_continue(0).is_null());
    assert!(anticipo_continue(257).is_null());
    let p = anticipo_continue(4);
    assert!(!p.is_null());
    let notes = unsafe { std::slice::from_raw_parts(p, 4) };
    assert!(notes.iter().all(|&n| n < 108));
}

#[test]
fn ffi_exposes_hidden_layers() {
    anticipo_init(3, 10, 0.1, 1);
    assert_eq!(anticipo_hidden(), 10);
    let read = || unsafe { std::slice::from_raw_parts(anticipo_layers(), 20).to_vec() };
    assert!(read().iter().all(|v| *v == 0.0), "zeros before any input");
    for _ in 0..30 {
        for b in b"abc" {
            anticipo_observe(*b as u32);
        }
    }
    let early_err: f32 = read()[10..].iter().map(|v| v * v).sum();
    anticipo_observe(b'z' as u32); // surprising symbol
    let l = read();
    assert!(l[..10].iter().all(|a| a.abs() <= 1.0));
    assert!(l[..10].iter().any(|a| a.abs() > 0.0));
    let surprise_err: f32 = l[10..].iter().map(|v| v * v).sum();
    assert!(surprise_err > early_err, "{surprise_err} vs {early_err}");
    anticipo_reset();
    assert!(read().iter().all(|v| *v == 0.0));
}

#[test]
fn state_round_trips_through_ffi() {
    anticipo_init(3, 16, 0.1, 7);
    for _ in 0..40 {
        for b in b"abc" {
            anticipo_observe(*b as u32);
        }
    }
    let trained = probs();
    let n = anticipo_state_export() as usize;
    assert!(n > 16);
    let saved = unsafe { std::slice::from_raw_parts(anticipo_state_ptr(), n) }.to_vec();

    // A fresh model of the same shape behaves differently until loaded.
    anticipo_init(3, 16, 0.1, 7);
    assert_ne!(probs(), trained);
    let dst = anticipo_state_buffer(n as u32);
    unsafe { std::ptr::copy_nonoverlapping(saved.as_ptr(), dst, n) };
    assert_eq!(anticipo_state_import(), 1);
    // Context was cleared on import, so replay the same context as before.
    for b in b"abc" {
        anticipo_observe(*b as u32);
    }
    let after = probs();
    assert!(after[b'a' as usize] > 0.9, "{}", after[b'a' as usize]);
}

#[test]
fn bad_state_is_rejected_and_model_unchanged() {
    let mut s = Session::new(3, 16, 0.1, 1);
    s.observe_all(&b"abcabc".repeat(10));
    let before = s.predict();
    let good = s.export_state();

    assert!(s.import_state(b"junk").is_err());
    assert!(s.import_state(&good[..good.len() - 4]).is_err());
    let mut nan = good.clone();
    nan[20..24].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(s.import_state(&nan).is_err());
    let mut other = Session::new(2, 16, 0.1, 1);
    let err = other.import_state(&good).unwrap_err();
    assert!(err.contains("shape mismatch"), "{err}");
    assert_eq!(s.predict(), before);

    // Import clears the context, so compare against a context-cleared source.
    s.reset_context();
    let before = s.predict();
    let mut twin = Session::new(3, 16, 0.1, 99);
    twin.import_state(&good).unwrap();
    assert_eq!(twin.predict(), before);
    assert_eq!(twin.export_state(), good);
}
