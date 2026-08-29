//! The pontine relay -- a *sibling* of the cerebellum, not a part of it.
//!
//! The pontine nuclei are brainstem. Their anatomical role is to sit between cortex and the
//! cerebellar mossy fibres, which is exactly this module's position in the data flow: it reads
//! shared memory, which the policy writes, and hands the result to `cerebellum::net`. Filing it
//! under `cerebellum/` would have put a structure inside the one it projects to.
//!
//! The path from the policy layer down to the cerebellum's mossy fibres.
//!
//! # What this does not do
//!
//! It does not compute. The pontine nuclei are a relay -- cortex projects to them, they project to
//! the mossy fibres, and the expansion into a separable code happens *after*, in the granule
//! layer. That is the whole reason this file is thirty lines of filtering instead of a network:
//! the divergence is already paid for by 16384 granule cells reading a fixed random projection, so
//! a layer here would be duplicating the one part of the design that is deliberately not learned.
//!
//! Putting a *trained* MLP here would cost more than duplication. The cerebellum learns with no
//! backward pass because there is exactly one layer of adjustable synapses and its error arrives
//! already in that layer's output units (PWM). A learned layer upstream of the granule code would
//! have to have its error routed back through the expansion and the readout to reach it -- which
//! is precisely the credit assignment this design does not have and does not want.
//!
//! # What it relays
//!
//! Identity, not mass. See [`crate::shm::NUM_CONTEXT`] for why the policy is not asked how heavy
//! the object is. The relay's whole job is to get that declaration onto the mossy fibres without
//! introducing a transient the reflex would have to absorb.
//!
//! # Why there is a filter here at all
//!
//! The two ends run at different rates: the policy publishes context at inference rate (~30 Hz)
//! and the cerebellum reads mossy fibres at 200 Hz. An unfiltered channel therefore steps, and a
//! step in the granule code is a step in the feedforward -- the readout is linear in the code, so
//! the arm would get a PWM discontinuity at the instant the policy changed its mind. The
//! feedforward is slew-limited downstream, but a slew limiter turns a step into a ramp at a fixed
//! rate regardless of how far it has to travel; a first-order lag makes the *code itself* move
//! continuously, so what the readout sees was always a state the network could have been in.

use crate::shm::NUM_CONTEXT;

/// Relays [`crate::shm::InputData::context`] onto the mossy fibres, low-passed.
pub struct PontineRelay {
    tau_s: f32,
    /// Set by `--cerebellum-context`, which replaces the shared-memory channel entirely.
    override_value: Option<[f32; NUM_CONTEXT]>,
    filtered: [f32; NUM_CONTEXT],
}

impl PontineRelay {
    pub fn new(tau_ms: u64, override_value: Option<[f32; NUM_CONTEXT]>) -> Self {
        Self {
            tau_s: tau_ms as f32 / 1000.0,
            override_value,
            // Zero is the no-context state, and it is also where an untrained network's
            // contribution is zero, so starting here means the first ticks add nothing rather
            // than adding something arbitrary.
            filtered: [0.0; NUM_CONTEXT],
        }
    }

    /// One tick. `from_policy` is the raw shared-memory value; the return is what the mossy fibres
    /// should carry.
    pub fn relay(&mut self, from_policy: [f32; NUM_CONTEXT], dt_s: f32) -> [f32; NUM_CONTEXT] {
        // The override is a bench instrument: it is the only way to run the context experiment
        // before any policy can label a demonstration, so it bypasses the filter as well. What it
        // declares is meant to be exactly what the network sees.
        if let Some(v) = self.override_value {
            self.filtered = v;
            return v;
        }

        // `dt / (tau + dt)` rather than `dt / tau`, so the step is bounded by 1 for any dt and a
        // long stall cannot make the filter overshoot on the tick that follows it.
        let alpha = if self.tau_s > 0.0 {
            dt_s / (self.tau_s + dt_s)
        } else {
            1.0
        };
        for (filtered, &x) in self.filtered.iter_mut().zip(from_policy.iter()) {
            // A non-finite sample is dropped rather than filtered. Feeding one in would make the
            // filter state non-finite permanently -- there is no later value that divides a NaN
            // back out -- so a single bad publish from the policy would disable the context
            // channel for the rest of the run. `net::implausible_input` still checks the result;
            // this is what keeps that check from becoming a permanent stall instead of a skipped
            // tick.
            if !x.is_finite() {
                continue;
            }
            *filtered += (x - *filtered) * alpha;
        }
        self.filtered
    }
}

/// Parses `--cerebellum-context`, e.g. `"1,0"`.
pub fn parse_context(spec: &str) -> Result<[f32; NUM_CONTEXT], String> {
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    if parts.len() != NUM_CONTEXT {
        return Err(format!(
            "expected {NUM_CONTEXT} comma-separated values, got {}",
            parts.len()
        ));
    }
    let mut out = [0f32; NUM_CONTEXT];
    for (i, p) in parts.iter().enumerate() {
        let v: f32 = p
            .parse()
            .map_err(|e| format!("`{p}` is not a number: {e}"))?;
        if !v.is_finite() {
            return Err(format!("`{p}` is not a finite value"));
        }
        out[i] = v;
    }
    Ok(out)
}
