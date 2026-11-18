//! C-ABI exports for the browser. No `wasm-bindgen`, no dependencies: JS
//! instantiates the module and talks to it through plain numbers plus typed
//! arrays viewed over linear memory (see `web/anticipo.js`).
//!
//! State lives in a thread-local session (WASM is single-threaded). Buffers
//! returned by pointer stay valid until the next call that writes them.

use crate::session::{Session, VOCAB};
use std::cell::RefCell;

struct State {
    session: Session,
    probs: Vec<f32>,
    input: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with_state<R>(default: R, f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|s| match s.borrow_mut().as_mut() {
        Some(st) => f(st),
        None => default,
    })
}

/// Create (or replace) the session. Must be called first.
#[no_mangle]
pub extern "C" fn anticipo_init(window: u32, hidden: u32, learning_rate: f32, seed: u32) {
    let session = Session::new(window as usize, hidden as usize, learning_rate, seed as u64);
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            probs: session.predict(),
            session,
            input: Vec::new(),
        })
    });
}

/// Alphabet size (length of the probability buffer).
#[no_mangle]
pub extern "C" fn anticipo_vocab() -> u32 {
    VOCAB as u32
}

/// Observe one ASCII byte; returns its surprise in nats (-1 if uninitialised).
#[no_mangle]
pub extern "C" fn anticipo_observe(byte: u32) -> f32 {
    with_state(-1.0, |st| {
        let step = st.session.observe(byte.min(255) as u8);
        st.probs = st.session.predict();
        step.surprise
    })
}

/// Pointer to `anticipo_vocab()` f32 probabilities for the next symbol
/// (null if uninitialised).
#[no_mangle]
pub extern "C" fn anticipo_predict() -> *const f32 {
    with_state(std::ptr::null(), |st| {
        st.probs = st.session.predict();
        st.probs.as_ptr()
    })
}

/// Reserve `len` bytes of input scratch space and return its address, so JS
/// can write UTF-8 there and call `anticipo_observe_buffer`.
#[no_mangle]
pub extern "C" fn anticipo_input_buffer(len: u32) -> *mut u8 {
    with_state(std::ptr::null_mut(), |st| {
        st.input.clear();
        st.input.resize(len as usize, 0);
        st.input.as_mut_ptr()
    })
}

/// Observe the first `len` bytes of the input buffer. Returns the mean
/// surprise in nats, or -1 if uninitialised, `len` is 0, or `len` exceeds
/// the buffer.
#[no_mangle]
pub extern "C" fn anticipo_observe_buffer(len: u32) -> f32 {
    with_state(-1.0, |st| {
        let len = len as usize;
        if len == 0 || len > st.input.len() {
            return -1.0;
        }
        let s = st.session.observe_all(&st.input[..len]);
        st.probs = st.session.predict();
        s.iter().sum::<f32>() / s.len() as f32
    })
}

/// Clear the context window, keep the weights.
#[no_mangle]
pub extern "C" fn anticipo_reset_context() {
    with_state((), |st| st.session.reset_context());
}

/// Forget everything the model has learned.
#[no_mangle]
pub extern "C" fn anticipo_reset() {
    with_state((), |st| {
        st.session.reset();
        st.probs = st.session.predict();
    });
}
