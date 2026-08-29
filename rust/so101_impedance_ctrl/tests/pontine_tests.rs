//! Tests for the pontine relay: the filter, the bench override, and the argument parsing.
//!
//! The relay has no math worth testing beyond a first-order lag. What it does have is three ways
//! to quietly ruin a run -- a step that the reflex has to absorb, a NaN that disables the channel
//! for the rest of the session, and a pinned context that is silently not the one that was typed
//! -- and those are what is pinned here.

use so101_impedance_ctrl::pontine::{parse_context, PontineRelay};
use so101_impedance_ctrl::shm::NUM_CONTEXT;

const DT: f32 = 1.0 / 400.0;

#[test]
fn the_filter_ramps_rather_than_steps() {
    // The policy publishes at inference rate and the cerebellum reads at 200 Hz. An unfiltered
    // channel would put a discontinuity in the granule code, and the readout is linear in that
    // code, so the arm would feel it as a PWM step.
    let mut relay = PontineRelay::new(100, None);
    let step = [1.0, 1.0];

    let first = relay.relay(step, DT)[0];
    assert!(
        first < 0.05,
        "one tick moved the context {first} of the way -- that is a step, not a ramp"
    );

    // And it must actually arrive: a lag that never converges is just a broken channel.
    let mut last = first;
    for _ in 0..400 {
        last = relay.relay(step, DT)[0];
    }
    assert!(
        (last - 1.0).abs() < 0.05,
        "after a second the context reached only {last}"
    );
    assert!(last <= 1.0 + 1e-6, "the lag overshot to {last}");
}

#[test]
fn a_long_stall_cannot_make_the_filter_overshoot() {
    // `alpha = dt / (tau + dt)` rather than `dt / tau`, so a tick that arrives late -- a blown
    // deadline, a resumed process -- moves the filter at most all the way, never past.
    let mut relay = PontineRelay::new(100, None);
    let out = relay.relay([1.0, 1.0], 10.0)[0];
    assert!(
        (0.0..=1.0).contains(&out),
        "a 10 s tick drove the filter to {out}"
    );
}

#[test]
fn a_non_finite_sample_is_dropped_rather_than_filtered() {
    // The failure this exists for: a filter state is a running sum, and no later value divides a
    // NaN back out. One bad publish would otherwise disable the context channel permanently --
    // `net::implausible_input` would refuse every subsequent tick, turning a skipped sample into a
    // dead feedforward.
    let mut relay = PontineRelay::new(100, None);
    for _ in 0..400 {
        relay.relay([1.0, 1.0], DT);
    }
    let good = relay.relay([1.0, 1.0], DT);

    let poisoned = relay.relay([f32::NAN, f32::INFINITY], DT);
    assert!(
        poisoned.iter().all(|v| v.is_finite()),
        "a non-finite sample reached the filter state: {poisoned:?}"
    );
    for i in 0..NUM_CONTEXT {
        assert!(
            (poisoned[i] - good[i]).abs() < 1e-6,
            "the relay should hold its last good value, got {} after {}",
            poisoned[i],
            good[i]
        );
    }
}

#[test]
fn the_bench_override_is_exactly_what_was_typed() {
    // The override bypasses the filter as well as shared memory. It exists so a person at a bench
    // can declare a context and know the network saw that value -- a ramp would mean the first
    // seconds of every trial were spent somewhere between the two contexts being compared.
    let mut relay = PontineRelay::new(100, Some([-1.0, 1.0]));
    assert_eq!(relay.relay([0.0, 0.0], DT), [-1.0, 1.0]);
    assert_eq!(relay.relay([0.5, 0.5], DT), [-1.0, 1.0]);
}

#[test]
fn a_zero_time_constant_passes_through() {
    let mut relay = PontineRelay::new(0, None);
    assert_eq!(relay.relay([1.0, -1.0], DT), [1.0, -1.0]);
}

#[test]
fn context_parsing_refuses_what_it_cannot_relay() {
    assert_eq!(parse_context("1,-1").unwrap(), [1.0, -1.0]);
    assert_eq!(parse_context(" 0.5 , 0 ").unwrap(), [0.5, 0.0]);

    // The wrong arity is the likely typo, and it must not be padded with zeros: `--cerebellum-context 1`
    // meaning `[1, 0]` would silently be the weak one-fibre encoding the help text warns about.
    assert!(parse_context("1").is_err());
    assert!(parse_context("1,0,0").is_err());
    assert!(parse_context("").is_err());
    assert!(parse_context("1,x").is_err());
    assert!(parse_context("1,nan").is_err(), "NaN is not a context");
    assert!(parse_context("1,inf").is_err());
}
