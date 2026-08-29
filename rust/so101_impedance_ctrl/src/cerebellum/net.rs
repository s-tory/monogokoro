//! The network itself: shapes, the fixed granule-layer connectivity, the mossy-fibre encoding,
//! and a CPU reference implementation of every stage.
//!
//! This module is the *definition* of the math. `gpu.rs` and `shaders/*.comp` are a second
//! implementation of exactly what is written here, and `tests/cerebellum_net_tests.rs` checks the
//! two agree numerically. That ordering is deliberate: a compute shader is not debuggable by
//! inspection, so the readable version has to be the authority, and the fast version has to be
//! held against it.
//!
//! # Layers
//!
//! ```text
//!   mossy fibres        32   proprioception + efference copy, plus 2 pontine context channels
//!        |  fixed sparse random projection, GC_FAN_IN inputs per cell, never learned
//!   granule cells    16384   tanh, then Golgi (global subtractive) inhibition -> sparse
//!        |  parallel fibres, with an eligibility trace
//!   Purkinje cells       6   linear readout -- the ONLY learned layer
//!        ^  climbing fibre: the reflex's own standing duty
//! ```
//!
//! # Why the expansion layer is not learned
//!
//! This is the whole reason no backpropagation is needed. Marr and Albus's argument is that a
//! large, sparse, *random* recoding of the input already separates patterns well enough that a
//! single linear readout on top can fit any smooth function of the input -- so the only thing that
//! has to be learned is that readout, and a linear readout is trained by a local rule that needs
//! no gradient flowing backwards through anything. The credit assignment problem that
//! backpropagation exists to solve simply does not arise, because there is exactly one layer of
//! adjustable synapses and the error signal is already expressed in that layer's output units
//! (PWM).
//!
//! The cost of that trick is parameters instead of depth: 16384 granule cells to get the
//! separation a trained hidden layer would get with a few hundred. On a CPU inside a 2.5 ms tick
//! that trade would be uncomfortable. On an iGPU it is nothing, which is what makes the whole
//! biologically-shaped design the *practical* choice here rather than a homage.

use crate::control::ENCODER_RESOLUTION;
use crate::shm::{NUM_CONTEXT, NUM_MOTORS};

/// Mossy-fibre count: 5 signals per joint (see [`encode_mossy_fibres`]).
pub const MF_PER_JOINT: usize = 5;

/// The proprioceptive block, which occupies the front of the mossy-fibre vector.
pub const MF_PROPRIOCEPTIVE: usize = NUM_MOTORS * MF_PER_JOINT;

/// Proprioception plus the pontine context channels (see [`crate::shm::NUM_CONTEXT`]).
///
/// The context sits at the tail rather than interleaved per joint because it is not a property of
/// any joint: it describes the situation the whole arm is in. Granule cells sample without regard
/// to position in this vector, so the layout is for readers, not for the network.
pub const MF_DIM: usize = MF_PROPRIOCEPTIVE + NUM_CONTEXT;

/// One Purkinje output per motor. The gripper's is computed but masked off before it reaches a
/// servo (see `mod.rs`) -- cheaper and simpler than a ragged layout, and it means the gripper's
/// column can be inspected without ever being applied.
pub const NUM_OUTPUTS: usize = NUM_MOTORS;

/// Mossy fibres sampled by each granule cell.
///
/// Four, because that is what a real granule cell gets, and because the reason it is four turns
/// out to be the reason it works: a small fan-in makes each cell respond to a *conjunction* of a
/// few inputs rather than to a projection of all of them, which is what produces the sparse,
/// weakly-correlated codes the readout needs. It is also why the expansion is cheap --
/// `gc_dim * 4` multiply-adds instead of `gc_dim * MF_DIM`.
pub const GC_FAN_IN: usize = 4;

/// Squashing scales for the mossy-fibre encoding, in the raw units the daemon works in.
///
/// Every channel is `tanh`-squashed after scaling, so these set where each signal's useful range
/// sits on the nonlinearity rather than acting as a hard limit. Positions are exempt: they are
/// encoded as a phase instead (see below).
const VEL_SCALE: f32 = 500.0; // counts/s
const ERR_SCALE: f32 = 100.0; // counts
const CURRENT_SCALE: f32 = 200.0; // raw Present_Current units

/// Multiples of a channel's own scale beyond which a sensory value cannot be believed.
///
/// Derived rather than tuned. Every squashed channel enters as `tanh(raw / scale)`, and `tanh`
/// reaches 1.0 in `f32` by about ten scales -- past that the network cannot tell one value from
/// another, so nothing is lost by refusing them. Twenty is double that point, which leaves every
/// legitimate reading untouched (`pos_error` is bounded by +/-2048 by construction, and 20 x
/// `ERR_SCALE` is 2000) while still sitting well inside the corruption this exists for: a raw
/// `Present_Current` word read as 32790 is 164 scales out.
///
/// The check runs on the raw snapshot, *before* the encoding, because the encoding is what
/// destroys the evidence -- through `tanh` a corrupted 32790 and a merely large 4000 are both
/// exactly 1.0.
pub const MF_SATURATION_FACTOR: f32 = 20.0;

/// The first sensory reading that could not have come from this arm, as `(channel, joint, value)`.
///
/// The cerebellum's inputs are the one part of this daemon with no plausibility check of their
/// own: a position is guarded by `--max-pos-slew`, and velocity and tracking error are computed
/// from positions, but `Present_Current` is read straight off the bus and believed. It is also
/// what broke -- an undecoded sign-magnitude word put six of the thirty mossy fibres on the rail
/// at once, and the readout, a linear function over a random projection, extrapolated a
/// feedforward to its clamp on a joint carrying no load.
///
/// Asking here rather than at the granule layer is deliberate, and it was measured: the Golgi loop
/// holds the granule code at a fixed sparsity whatever the input, so a corrupted state produces a
/// perfectly ordinary-looking code. Scaling the readout by how much of that code had been taught
/// gave 0.994 for the corruption against 0.99999 for a normal pose -- no separation at all,
/// because the same excitable cells win the threshold competition for almost any input. The
/// anomaly is visible in the input and normalised away by the time it reaches the cells.
pub fn implausible_input(state: &SensoryState) -> Option<(&'static str, usize, f32)> {
    let k = MF_SATURATION_FACTOR;
    for j in 0..NUM_MOTORS {
        let checks = [
            ("velocity", state.present_vel[j], k * VEL_SCALE),
            ("tracking error", state.pos_error[j], k * ERR_SCALE),
            ("current", state.present_current[j], k * CURRENT_SCALE),
        ];
        for (name, value, limit) in checks {
            if !value.is_finite() || value.abs() > limit {
                return Some((name, j, value));
            }
        }
        // Position is not squashed -- it enters as a phase, which is defined for any number. The
        // bound is the encoder instead: outside its range the reading is not a pose at all.
        let pos = state.present_pos[j];
        if !pos.is_finite() || !(0.0..ENCODER_RESOLUTION).contains(&pos) {
            return Some(("position", j, pos));
        }
    }
    // Context is refused for being unrepresentable, not for being large. Out of range it is
    // clamped in the encoding: a policy that overshoots its own [-1, 1] contract is still saying
    // something meaningful about which situation the arm is in, and stalling the whole feedforward
    // over it would trade a small error for a total one. NaN says nothing, and unlike a big number
    // it poisons every granule cell that draws it.
    for i in 0..NUM_CONTEXT {
        if !state.context[i].is_finite() {
            return Some(("context", i, state.context[i]));
        }
    }
    None
}

/// The sensory snapshot the RT loop hands to the cerebellum each tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct SensoryState {
    pub present_pos: [f32; NUM_MOTORS],
    pub present_vel: [f32; NUM_MOTORS],
    /// `wrapped_delta(target, present)` -- the reflex's own error, i.e. efference copy.
    pub pos_error: [f32; NUM_MOTORS],
    pub present_current: [f32; NUM_MOTORS],
    /// Feedback duty the impedance law is producing. This is the climbing-fibre signal.
    pub fb_pwm: [f32; NUM_MOTORS],
    /// Pontine context relayed from the policy layer, nominally in `[-1, 1]`. All-zero is the
    /// no-context case and contributes nothing.
    pub context: [f32; NUM_CONTEXT],
}

/// Builds the mossy-fibre vector from one sensory snapshot.
///
/// Position enters as `(sin, cos)` of the encoder phase rather than as a number. That is not
/// decoration: `Present_Position` wraps 4095 -> 0 mid-travel, and `wrist_roll` is calibrated over
/// the full turn so no choice of `Homing_Offset` can put the seam somewhere it never visits (the
/// same fact that forces `wrapped_delta` to exist). A raw count would make the network see a
/// full-scale jump where the joint moved one count, and it would learn a feedforward with a cliff
/// in it. A phase pair is continuous everywhere and costs one extra input per joint.
pub fn encode_mossy_fibres(state: &SensoryState) -> [f32; MF_DIM] {
    let mut mf = [0f32; MF_DIM];
    for j in 0..NUM_MOTORS {
        let phase = std::f32::consts::TAU * state.present_pos[j] / ENCODER_RESOLUTION;
        let base = j * MF_PER_JOINT;
        mf[base] = phase.sin();
        mf[base + 1] = phase.cos();
        mf[base + 2] = (state.present_vel[j] / VEL_SCALE).tanh();
        mf[base + 3] = (state.pos_error[j] / ERR_SCALE).tanh();
        mf[base + 4] = (state.present_current[j] / CURRENT_SCALE).tanh();
    }
    // The pontine channels enter as they are, only clamped. Every other channel is squashed
    // because it arrives in physical units whose useful range has to be mapped onto the
    // nonlinearity; context arrives already normalised by contract, and squashing a declared
    // 1.0 down to 0.76 would only make the two contexts harder for the expansion to separate.
    for i in 0..NUM_CONTEXT {
        mf[MF_PROPRIOCEPTIVE + i] = state.context[i].clamp(-1.0, 1.0);
    }
    mf
}

/// The fixed (never-learned) granule-layer connectivity.
///
/// Generated from a seed rather than stored, so the same `--cerebellum-seed` reproduces the same
/// network on any machine. That matters for the weights file: Purkinje weights are only meaningful
/// against the granule code that produced them, so a saved weight set is bound to its seed and
/// `gc_dim` and is refused if either changed.
pub struct GranuleParams {
    /// `gc_dim * GC_FAN_IN` mossy-fibre indices.
    pub idx: Vec<u32>,
    /// `gc_dim * GC_FAN_IN` synaptic weights.
    pub weight: Vec<f32>,
    /// `gc_dim` per-cell thresholds, which give cells different operating points so they do not
    /// all switch at once.
    pub bias: Vec<f32>,
    pub gc_dim: usize,
    pub seed: u64,
}

/// SplitMix64. A dependency-free, well-tested, deterministic generator -- these draws must be
/// byte-identical across machines and across builds for a saved weight file to stay valid, which
/// rules out anything seeded from the environment.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Roughly standard-normal, by summing 12 uniforms (Irwin-Hall). Exact enough for initialising
    /// a random projection, and it avoids pulling in a distributions crate.
    fn next_normal(&mut self) -> f32 {
        (0..12).map(|_| self.next_f32()).sum::<f32>() - 6.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

impl GranuleParams {
    pub fn generate(gc_dim: usize, seed: u64) -> Self {
        assert!(gc_dim > 0, "granule layer must have at least one cell");
        let mut rng = SplitMix64(seed);
        let mut idx = Vec::with_capacity(gc_dim * GC_FAN_IN);
        let mut weight = Vec::with_capacity(gc_dim * GC_FAN_IN);
        let mut bias = Vec::with_capacity(gc_dim);

        for _ in 0..gc_dim {
            // Sample GC_FAN_IN *distinct* mossy fibres. Duplicates would silently turn a
            // conjunction of four signals into a scaled response to three, which is a quiet way to
            // lose diversity in exactly the layer whose only job is to provide it.
            let mut chosen = [u32::MAX; GC_FAN_IN];
            for k in 0..GC_FAN_IN {
                loop {
                    let candidate = rng.below(MF_DIM as u64) as u32;
                    if !chosen[..k].contains(&candidate) {
                        chosen[k] = candidate;
                        break;
                    }
                }
            }
            idx.extend_from_slice(&chosen);
            // Scaled so the pre-activation has roughly unit variance for unit-variance inputs,
            // which puts cells on the useful part of tanh instead of saturated at one end.
            let scale = 1.0 / (GC_FAN_IN as f32).sqrt();
            for _ in 0..GC_FAN_IN {
                weight.push(rng.next_normal() * scale);
            }
            bias.push(rng.next_normal() * 0.5);
        }

        Self {
            idx,
            weight,
            bias,
            gc_dim,
            seed,
        }
    }
}

/// Granule-cell activations for one mossy-fibre vector, *before* Golgi inhibition.
///
/// Kept separate from the thresholding step so tests can look at the raw distribution, which is
/// what the Golgi threshold is calibrated against.
pub fn granule_preactivation(params: &GranuleParams, mf: &[f32; MF_DIM], out: &mut [f32]) {
    debug_assert_eq!(out.len(), params.gc_dim);
    for (j, slot_out) in out.iter_mut().enumerate() {
        let mut sum = params.bias[j];
        for k in 0..GC_FAN_IN {
            let slot = j * GC_FAN_IN + k;
            sum += params.weight[slot] * mf[params.idx[slot] as usize];
        }
        *slot_out = sum.tanh();
    }
}

/// Golgi-cell inhibition: a single global subtractive threshold, then rectification.
///
/// Returns the fraction of cells left active, which is what the caller's threshold integrator
/// consumes.
///
/// A global scalar is the entire mechanism, and it is enough: Golgi cells pool the parallel-fibre
/// output and inhibit the whole granule field in proportion, which is a negative feedback loop on
/// *total activity*. Implementing it as feedback rather than as a k-winners-take-all sort is both
/// closer to the biology and much better suited to a GPU -- no sort, no scan, one atomic counter,
/// and the threshold converges to whatever value yields the target sparsity without anyone having
/// to assume the activations are normally distributed.
pub fn golgi_inhibit(pre: &[f32], theta: f32, out: &mut [f32]) -> f32 {
    debug_assert_eq!(pre.len(), out.len());
    let mut active = 0usize;
    for (o, &p) in out.iter_mut().zip(pre.iter()) {
        let v = p - theta;
        if v > 0.0 {
            *o = v;
            active += 1;
        } else {
            *o = 0.0;
        }
    }
    active as f32 / pre.len().max(1) as f32
}

/// Rescales the granule code to unit L2 norm, returning the scale that was applied.
///
/// Without this the effective learning rate is `rate * sum_j g_j^2`, which varies with `gc_dim`,
/// with whatever sparsity the Golgi loop settled at, and with how far above threshold the active
/// cells sit -- so `--cerebellum-rate` would mean something different for every layer size, by
/// more than an order of magnitude between plausible settings. With `|g| = 1` it means exactly one
/// thing: the fraction of the remaining error corrected per step, giving a time constant of
/// `1 / rate` steps regardless of how the layer is configured.
///
/// It also makes the readout an interpolation between the values stored for nearby granule codes
/// rather than an unnormalised sum, which is the same property a normalised radial-basis network
/// gets by dividing through by its total activation.
pub fn normalise_activity(gc: &mut [f32]) -> f32 {
    let total: f32 = gc.iter().map(|g| g * g).sum();
    // A silenced layer (threshold above every cell) scales to zero, not to infinity.
    if total <= 1e-12 {
        return 0.0;
    }
    let inv = total.sqrt().recip();
    for g in gc.iter_mut() {
        *g *= inv;
    }
    inv
}

/// Purkinje readout: `ff[i] = sum_j W[i][j] * gc[j]`.
///
/// `weights` is row-major `[NUM_OUTPUTS][gc_dim]`.
pub fn purkinje_readout(weights: &[f32], gc: &[f32], gc_dim: usize) -> [f32; NUM_OUTPUTS] {
    let mut ff = [0f32; NUM_OUTPUTS];
    for (i, slot) in ff.iter_mut().enumerate() {
        let row = &weights[i * gc_dim..(i + 1) * gc_dim];
        *slot = row.iter().zip(gc.iter()).map(|(w, g)| w * g).sum();
    }
    ff
}

/// Decays the eligibility trace toward the current parallel-fibre activity.
///
/// The trace exists because the climbing-fibre signal is *late*. What the reflex is straining
/// against right now is the consequence of the pose the arm was in some tens of milliseconds ago
/// -- serial round trips, the velocity filter's group delay, the low-pass on the error signal, and
/// the arm's own mechanical response all sit between the two. Pairing today's error with today's
/// granule code would credit the wrong cells. A trace with a time constant that covers that delay
/// pairs the error with everything that was recently active instead, which is exactly the role of
/// the ~100-200 ms eligibility window at the parallel-fibre synapse.
pub fn decay_trace(trace: &mut [f32], gc: &[f32], decay: f32) {
    debug_assert_eq!(trace.len(), gc.len());
    for (t, &g) in trace.iter_mut().zip(gc.iter()) {
        *t = *t * decay + g * (1.0 - decay);
    }
}

/// The learning rule: three-factor Hebbian with heterosynaptic decay.
///
/// ```text
/// dW[i][j] = rate * ( cf[i] * e[j]  -  leak * W[i][j] * e[j] )
///                     \___________/     \__________________/
///                      Hebbian: pre x    heterosynaptic LTD: the "modified" half.
///                      post-error        Without it a plain Hebbian product only ever
///                                        grows, because nothing in it is negative-going
///                                        once the sign of cf is consistent. Making the
///                                        decay proportional to e[j] means a synapse
///                                        forgets only while it is being used, so the
///                                        parts of the map the arm is not visiting keep
///                                        what they learned.
/// ```
///
/// The sign is positive on purpose. In the cerebellum the climbing fibre depresses co-active
/// parallel-fibre synapses, and the Purkinje cell is inhibitory onto the deep nuclei, so two sign
/// inversions sit between that LTD and the motor command. This layer stands in for the whole
/// PC -> DCN path, so the composition is what gets implemented: a standing feedback duty in one
/// direction has to grow the feedforward in the *same* direction, or the loop is positive feedback
/// rather than compensation.
pub fn hebbian_update(
    weights: &mut [f32],
    trace: &[f32],
    cf: &[f32; NUM_OUTPUTS],
    gc_dim: usize,
    rate: f32,
    leak: f32,
) {
    for i in 0..NUM_OUTPUTS {
        let row = &mut weights[i * gc_dim..(i + 1) * gc_dim];
        let cf_i = cf[i];
        for (w, &e) in row.iter_mut().zip(trace.iter()) {
            if e == 0.0 {
                continue; // silent parallel fibre: no eligibility, no plasticity
            }
            *w += rate * (cf_i * e - leak * *w * e);
        }
    }
}

/// CPU reference implementation of one full cerebellar step.
///
/// Used by the tests as ground truth for the shaders, and available at runtime as
/// `--cerebellum-backend cpu` for bringing the daemon up on a machine with no working Vulkan
/// driver. It is not a silent fallback: if the GPU backend was asked for and fails, the daemon
/// says so and runs without a cerebellum rather than quietly switching, because "the feedforward
/// is running" and "the feedforward is running somewhere else at a different speed" are things an
/// operator needs told apart.
pub struct CpuNet {
    pub params: GranuleParams,
    pub weights: Vec<f32>,
    pre: Vec<f32>,
    gc: Vec<f32>,
    trace: Vec<f32>,
}

impl CpuNet {
    pub fn new(params: GranuleParams) -> Self {
        let gc_dim = params.gc_dim;
        Self {
            params,
            weights: vec![0.0; NUM_OUTPUTS * gc_dim],
            pre: vec![0.0; gc_dim],
            gc: vec![0.0; gc_dim],
            trace: vec![0.0; gc_dim],
        }
    }

    pub fn gc_dim(&self) -> usize {
        self.params.gc_dim
    }

    /// Runs inference and, if `cf` is `Some`, one learning step. Returns the feedforward duty and
    /// the fraction of granule cells that fired.
    pub fn step(
        &mut self,
        mf: &[f32; MF_DIM],
        theta: f32,
        trace_decay: f32,
        cf: Option<&[f32; NUM_OUTPUTS]>,
        rate: f32,
        leak: f32,
    ) -> ([f32; NUM_OUTPUTS], f32) {
        granule_preactivation(&self.params, mf, &mut self.pre);
        let active = golgi_inhibit(&self.pre, theta, &mut self.gc);
        normalise_activity(&mut self.gc);
        let ff = purkinje_readout(&self.weights, &self.gc, self.params.gc_dim);
        decay_trace(&mut self.trace, &self.gc, trace_decay);
        if let Some(cf) = cf {
            hebbian_update(
                &mut self.weights,
                &self.trace,
                cf,
                self.params.gc_dim,
                rate,
                leak,
            );
        }
        (ff, active)
    }

    pub fn trace(&self) -> &[f32] {
        &self.trace
    }
}
