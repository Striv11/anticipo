//! C-ABI exports for the browser. No `wasm-bindgen`, no dependencies: JS
//! instantiates the module and talks to it through plain numbers plus typed
//! arrays viewed over linear memory (see `web/anticipo.js`).
//!
//! State lives in a thread-local session (WASM is single-threaded). Buffers
//! returned by pointer stay valid until the next call that writes them.

use crate::session::Session;
use std::cell::RefCell;

struct State {
    session: Session,
    probs: Vec<f32>,
    input: Vec<u8>,
    notes: Vec<u32>,
    layers: Vec<f32>,
    state: Vec<u8>,
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
    install(Session::new(
        window as usize,
        hidden as usize,
        learning_rate,
        seed as u64,
    ));
}

/// Like `anticipo_init`, but a melody session: symbols are pitch ids
/// (`C0 = 0`; MIDI note number = id + 12), vocabulary 108.
#[no_mangle]
pub extern "C" fn anticipo_init_melody(window: u32, hidden: u32, learning_rate: f32, seed: u32) {
    install(Session::melody(
        window as usize,
        hidden as usize,
        learning_rate,
        seed as u64,
    ));
}

fn install(session: Session) {
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            probs: session.predict(),
            session,
            input: Vec::new(),
            notes: Vec::new(),
            layers: Vec::new(),
            state: Vec::new(),
        })
    });
}

/// Observe one symbol id (any mode); returns its surprise in nats, or -1 if
/// uninitialised or the id is outside the alphabet.
#[no_mangle]
pub extern "C" fn anticipo_observe_symbol(symbol: u32) -> f32 {
    with_state(-1.0, |st| {
        match st.session.observe_symbol(symbol as usize) {
            Some(step) => {
                st.probs = st.session.predict();
                step.surprise
            }
            None => -1.0,
        }
    })
}

/// Imagine the next `n` symbols (greedy; the model is not changed). Returns
/// a pointer to `n` u32 ids, or null if uninitialised or `n` is 0 or > 256.
#[no_mangle]
pub extern "C" fn anticipo_continue(n: u32) -> *const u32 {
    with_state(std::ptr::null(), |st| {
        if n == 0 || n > 256 {
            return std::ptr::null();
        }
        st.notes = st
            .session
            .continuation(n as usize)
            .into_iter()
            .map(|v| v as u32)
            .collect();
        st.notes.as_ptr()
    })
}

/// Alphabet size (length of the probability buffer).
#[no_mangle]
pub extern "C" fn anticipo_vocab() -> u32 {
    with_state(crate::session::VOCAB as u32, |st| st.session.vocab() as u32)
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

/// Hidden layer width (length of each layer buffer).
#[no_mangle]
pub extern "C" fn anticipo_hidden() -> u32 {
    with_state(0, |st| st.session.hidden() as u32)
}

/// Pointer to `2 * anticipo_hidden()` f32s describing the last observation:
/// first the hidden activity `tanh(x1)`, then the per-unit hidden prediction
/// errors. All zeros before anything has been observed; null if
/// uninitialised. Valid until the next call to this function.
#[no_mangle]
pub extern "C" fn anticipo_layers() -> *const f32 {
    with_state(std::ptr::null(), |st| {
        let h = st.session.hidden();
        st.layers.clear();
        match st.session.last_step() {
            Some(step) => {
                st.layers.extend_from_slice(&step.hidden_activity);
                st.layers.extend_from_slice(&step.hidden_errors);
            }
            None => st.layers.resize(2 * h, 0.0),
        }
        st.layers.as_ptr()
    })
}

/// Serialise the learned weights into an internal buffer and return its
/// length in bytes (0 if uninitialised). Read it via `anticipo_state_ptr`.
#[no_mangle]
pub extern "C" fn anticipo_state_export() -> u32 {
    with_state(0, |st| {
        st.state = st.session.export_state();
        st.state.len() as u32
    })
}

/// Pointer to the bytes written by the last `anticipo_state_export`.
#[no_mangle]
pub extern "C" fn anticipo_state_ptr() -> *const u8 {
    with_state(std::ptr::null(), |st| st.state.as_ptr())
}

/// Reserve `len` bytes of state scratch space (for JS to fill) and return
/// its address; follow with `anticipo_state_import`.
#[no_mangle]
pub extern "C" fn anticipo_state_buffer(len: u32) -> *mut u8 {
    with_state(std::ptr::null_mut(), |st| {
        st.state.clear();
        st.state.resize(len as usize, 0);
        st.state.as_mut_ptr()
    })
}

/// Load the state placed in the buffer from `anticipo_state_buffer`.
/// Returns 1 on success, 0 if uninitialised or the state is invalid or
/// does not match the current model shape (the model is then unchanged).
#[no_mangle]
pub extern "C" fn anticipo_state_import() -> u32 {
    with_state(0, |st| {
        let bytes = std::mem::take(&mut st.state);
        let ok = st.session.import_state(&bytes).is_ok();
        st.probs = st.session.predict();
        ok as u32
    })
}
