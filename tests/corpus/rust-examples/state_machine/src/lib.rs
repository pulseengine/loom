//! 4-state finite state machine kernel: Idle → Running → Paused → Done.
//!
//! Each transition is keyed on an integer event tag. Invalid transitions
//! return the same state (no-op). The exported entry point is `step(state,
//! event) -> next_state`, suitable for a tight wasm dispatch loop.
//!
//! In addition to the core `step` / `run` / `is_terminal` entry points, the
//! module exposes a small histogram helper that traces a run and emits the
//! per-state visit counts into a caller-provided buffer. The histogram code
//! exists primarily to give the LOOM optimizer a non-trivial code section
//! to chew on; it is not the kernel's primary role.

#![no_std]
#![allow(clippy::missing_safety_doc)]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq)]
enum State {
    Idle = 0,
    Running = 1,
    Paused = 2,
    Done = 3,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, Eq)]
enum Event {
    Start = 0,
    Pause = 1,
    Resume = 2,
    Stop = 3,
    Reset = 4,
    Tick = 5,
    Fault = 6,
    Recover = 7,
}

#[inline(always)]
fn decode_state(v: u32) -> Option<State> {
    match v {
        0 => Some(State::Idle),
        1 => Some(State::Running),
        2 => Some(State::Paused),
        3 => Some(State::Done),
        _ => None,
    }
}

#[inline(always)]
fn decode_event(v: u32) -> Option<Event> {
    match v {
        0 => Some(Event::Start),
        1 => Some(Event::Pause),
        2 => Some(Event::Resume),
        3 => Some(Event::Stop),
        4 => Some(Event::Reset),
        5 => Some(Event::Tick),
        6 => Some(Event::Fault),
        7 => Some(Event::Recover),
        _ => None,
    }
}

#[inline]
fn transition(s: State, e: Event) -> State {
    match (s, e) {
        (State::Idle, Event::Start) => State::Running,
        (State::Idle, Event::Reset) => State::Idle,
        (State::Idle, Event::Tick) => State::Idle,
        (State::Idle, Event::Fault) => State::Done,

        (State::Running, Event::Pause) => State::Paused,
        (State::Running, Event::Stop) => State::Done,
        (State::Running, Event::Tick) => State::Running,
        (State::Running, Event::Reset) => State::Idle,
        (State::Running, Event::Fault) => State::Done,

        (State::Paused, Event::Resume) => State::Running,
        (State::Paused, Event::Stop) => State::Done,
        (State::Paused, Event::Reset) => State::Idle,
        (State::Paused, Event::Tick) => State::Paused,
        (State::Paused, Event::Fault) => State::Done,

        (State::Done, Event::Reset) => State::Idle,
        (State::Done, Event::Recover) => State::Idle,
        (State::Done, _) => State::Done,

        // Any other combination is a no-op.
        _ => s,
    }
}

/// Encode an `(state, event)` pair into a single u8 tag. Used by the trace
/// helper below. Provides another set of branches for the wasm code section.
fn encode_pair(s: State, e: Event) -> u8 {
    let s_bits = s as u32;
    let e_bits = e as u32;
    ((s_bits << 4) | (e_bits & 0x0F)) as u8
}

/// Single-step transition. Returns the new state encoded as u32. Invalid
/// state or event values return `u32::MAX` to signal "bad input".
#[no_mangle]
pub extern "C" fn step(state: u32, event: u32) -> u32 {
    let s = match decode_state(state) {
        Some(s) => s,
        None => return u32::MAX,
    };
    let e = match decode_event(event) {
        Some(e) => e,
        None => return u32::MAX,
    };
    transition(s, e) as u32
}

/// Replay a sequence of events from `Idle`. Returns the final state, or
/// `u32::MAX` if any event in the sequence is invalid.
#[no_mangle]
pub unsafe extern "C" fn run(events_ptr: *const u32, events_len: usize) -> u32 {
    if events_ptr.is_null() && events_len != 0 {
        return u32::MAX;
    }
    let events = core::slice::from_raw_parts(events_ptr, events_len);
    let mut s = State::Idle;
    let mut i = 0usize;
    while i < events.len() {
        let e = match decode_event(events[i]) {
            Some(e) => e,
            None => return u32::MAX,
        };
        s = transition(s, e);
        i += 1;
    }
    s as u32
}

/// Return 1 if the state is terminal (`Done`), 0 otherwise. Used by callers
/// to short-circuit dispatch loops.
#[no_mangle]
pub extern "C" fn is_terminal(state: u32) -> u32 {
    match decode_state(state) {
        Some(State::Done) => 1,
        Some(_) => 0,
        None => u32::MAX,
    }
}

/// Replay a sequence of events from `start_state`, writing per-step state
/// visit counts (4 u32s, indexed by State as u32) into `counts`. Returns
/// the final state, or `u32::MAX` on bad input.
#[no_mangle]
pub unsafe extern "C" fn trace(
    start_state: u32,
    events_ptr: *const u32,
    events_len: usize,
    counts_ptr: *mut u32,
) -> u32 {
    let mut s = match decode_state(start_state) {
        Some(s) => s,
        None => return u32::MAX,
    };
    if events_ptr.is_null() && events_len != 0 {
        return u32::MAX;
    }
    if counts_ptr.is_null() {
        return u32::MAX;
    }
    // Zero the counts buffer (4 u32 slots).
    let mut k = 0usize;
    while k < 4 {
        core::ptr::write(counts_ptr.add(k), 0);
        k += 1;
    }
    let events = core::slice::from_raw_parts(events_ptr, events_len);
    let mut i = 0usize;
    while i < events.len() {
        let e = match decode_event(events[i]) {
            Some(e) => e,
            None => return u32::MAX,
        };
        s = transition(s, e);
        let idx = s as u32 as usize;
        let prev = core::ptr::read(counts_ptr.add(idx));
        core::ptr::write(counts_ptr.add(idx), prev.wrapping_add(1));
        i += 1;
    }
    s as u32
}

/// Compress a sequence of (state, event) pairs into a packed buffer of
/// u8 tags. Returns the number of bytes written. Returns `u32::MAX` if any
/// input is invalid or the output buffer is too small.
#[no_mangle]
pub unsafe extern "C" fn pack(
    states_ptr: *const u32,
    events_ptr: *const u32,
    len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
) -> u32 {
    if (states_ptr.is_null() || events_ptr.is_null() || out_ptr.is_null()) && len != 0 {
        return u32::MAX;
    }
    if len > out_cap {
        return u32::MAX;
    }
    let states = core::slice::from_raw_parts(states_ptr, len);
    let events = core::slice::from_raw_parts(events_ptr, len);
    let mut i = 0usize;
    while i < len {
        let s = match decode_state(states[i]) {
            Some(s) => s,
            None => return u32::MAX,
        };
        let e = match decode_event(events[i]) {
            Some(e) => e,
            None => return u32::MAX,
        };
        let tag = encode_pair(s, e);
        core::ptr::write(out_ptr.add(i), tag);
        i += 1;
    }
    len as u32
}

/// Return a 32-bit "fingerprint" of an event sequence by replaying it from
/// `Idle` and folding each visited state into a rolling hash. This is an
/// FNV-1a-ish mix purely to add arithmetic work to the wasm code section.
#[no_mangle]
pub unsafe extern "C" fn fingerprint(events_ptr: *const u32, events_len: usize) -> u32 {
    if events_ptr.is_null() && events_len != 0 {
        return u32::MAX;
    }
    let events = core::slice::from_raw_parts(events_ptr, events_len);
    let mut s = State::Idle;
    let mut h: u32 = 0x811C_9DC5;
    let mut i = 0usize;
    while i < events.len() {
        let e = match decode_event(events[i]) {
            Some(e) => e,
            None => return u32::MAX,
        };
        s = transition(s, e);
        h ^= s as u32;
        h = h.wrapping_mul(0x0100_0193);
        h ^= (events[i] & 0xFF) ^ 0xA5;
        h = h.wrapping_mul(0x0100_0193);
        i += 1;
    }
    h
}
