//! Tests for the cerebellum: the network math, the learning rule's actual behaviour, weight
//! persistence, and -- when a Vulkan device is present -- agreement between the shaders and the
//! CPU reference they were written against.

use so101_impedance_ctrl::cerebellum::net::{
    encode_mossy_fibres, golgi_inhibit, granule_preactivation, CpuNet, GranuleParams, SensoryState,
    GC_FAN_IN, MF_DIM, MF_PER_JOINT, NUM_OUTPUTS,
};
use so101_impedance_ctrl::cerebellum::net::{implausible_input, MF_SATURATION_FACTOR};
use so101_impedance_ctrl::cerebellum::{load_weights, parse_joints, save_weights};
use so101_impedance_ctrl::shm::NUM_MOTORS;

const TEST_GC_DIM: usize = 2048;
const TEST_SEED: u64 = 0x1234_5678_9ABC_DEF0;

fn test_params() -> GranuleParams {
    GranuleParams::generate(TEST_GC_DIM, TEST_SEED)
}

fn state_at(pos: f32) -> SensoryState {
    let mut s = SensoryState::default();
    for j in 0..NUM_MOTORS {
        s.present_pos[j] = pos + j as f32 * 37.0;
    }
    s
}

#[test]
fn mossy_fibre_encoding_is_continuous_across_the_encoder_wrap() {
    // The whole reason position enters as a phase pair: 4095 and 0 are one count apart on the
    // hardware, and the network must see them that way. Encoded as a raw count they would differ
    // by full scale, and the feedforward would have a cliff exactly where wrist_roll lives.
    let a = encode_mossy_fibres(&state_at(4095.0));
    let b = encode_mossy_fibres(&state_at(0.0));
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < 0.01,
            "encoding jumps across the wrap: {x} vs {y}"
        );
    }

    // A genuinely distant pose must, of course, still look different.
    let far = encode_mossy_fibres(&state_at(1024.0));
    assert!(
        a.iter().zip(far.iter()).any(|(x, y)| (x - y).abs() > 0.5),
        "encoding failed to distinguish a quarter-turn"
    );
}

#[test]
fn mossy_fibre_encoding_is_bounded() {
    // Every channel is either a sine/cosine or tanh-squashed, so absurd inputs must not be able to
    // hand the granule layer an enormous pre-activation.
    let s = SensoryState {
        present_vel: [1e6; NUM_MOTORS],
        pos_error: [-1e6; NUM_MOTORS],
        present_current: [1e6; NUM_MOTORS],
        ..SensoryState::default()
    };
    for v in encode_mossy_fibres(&s) {
        assert!(v.abs() <= 1.0 + 1e-6, "unbounded mossy fibre value {v}");
    }
}

#[test]
fn mossy_fibre_layout_covers_every_joint() {
    assert_eq!(MF_DIM, NUM_MOTORS * MF_PER_JOINT);
}

#[test]
fn granule_connectivity_is_deterministic_and_non_degenerate() {
    let a = test_params();
    let b = test_params();
    assert_eq!(a.idx, b.idx, "the same seed must produce the same network");
    assert_eq!(a.weight, b.weight);

    let different = GranuleParams::generate(TEST_GC_DIM, TEST_SEED ^ 1);
    assert_ne!(a.idx, different.idx, "a different seed must differ");

    // Each cell samples GC_FAN_IN *distinct* mossy fibres. Duplicates would quietly reduce a
    // conjunction of four signals to a scaled response to three.
    for cell in a.idx.chunks_exact(GC_FAN_IN) {
        for i in 0..GC_FAN_IN {
            assert!((cell[i] as usize) < MF_DIM);
            for j in (i + 1)..GC_FAN_IN {
                assert_ne!(cell[i], cell[j], "granule cell samples one fibre twice");
            }
        }
    }
}

#[test]
fn golgi_inhibition_rectifies_and_reports_sparsity() {
    let pre = vec![-0.5, 0.0, 0.25, 0.75, 1.0];
    let mut out = vec![0.0; pre.len()];
    let active = golgi_inhibit(&pre, 0.5, &mut out);
    assert_eq!(out, vec![0.0, 0.0, 0.0, 0.25, 0.5]);
    assert!((active - 2.0 / 5.0).abs() < 1e-6);

    // A threshold above everything silences the layer entirely, which is the state the integrator
    // has to be able to back out of.
    let active = golgi_inhibit(&pre, 2.0, &mut out);
    assert_eq!(active, 0.0);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn golgi_threshold_integrator_reaches_the_target_sparsity() {
    // The threshold is not computed from an assumed distribution -- it is a feedback loop on the
    // measured active fraction. This is the test that the loop actually closes.
    let params = test_params();
    let mut pre = vec![0.0; TEST_GC_DIM];
    let mut gc = vec![0.0; TEST_GC_DIM];
    let mf = encode_mossy_fibres(&state_at(1500.0));
    granule_preactivation(&params, &mf, &mut pre);

    let target = 0.02;
    let mut theta = 0.0f32;
    let mut active = 1.0;
    for _ in 0..500 {
        active = golgi_inhibit(&pre, theta, &mut gc);
        theta += 0.05 * (active - target);
        theta = theta.clamp(-1.0, 1.0);
    }
    assert!(
        (active - target).abs() < 0.01,
        "sparsity settled at {active}, wanted {target} (theta = {theta})"
    );
}

/// Runs the closed loop the daemon runs: a joint holds a constant load, the reflex produces the
/// duty needed to hold it, and that duty is the climbing-fibre signal.
///
/// Returns the feedback duty over time for joint 0.
fn run_load_cancellation(gc_dim: usize, steps: usize, rate: f32, leak: f32) -> Vec<f32> {
    let mut net = CpuNet::new(GranuleParams::generate(gc_dim, TEST_SEED));
    let load = 120.0f32; // the standing duty a loaded joint needs
    let mut theta = 0.0f32;
    let mut ff = [0f32; NUM_OUTPUTS];
    let mut history = Vec::with_capacity(steps);

    let mf = encode_mossy_fibres(&state_at(900.0));
    for _ in 0..steps {
        // The reflex supplies whatever the feedforward does not.
        let mut cf = [0f32; NUM_OUTPUTS];
        cf[0] = load - ff[0];
        history.push(cf[0]);

        let (out, active) = net.step(&mf, theta, 0.0, Some(&cf), rate, leak);
        ff = out;
        theta += 0.05 * (active - 0.02);
        theta = theta.clamp(-1.0, 1.0);
    }
    history
}

#[test]
fn hebbian_learning_takes_over_a_standing_load() {
    // The single functional claim the whole module rests on: with the feedback duty as the
    // teaching signal, the feedforward grows until the feedback duty is no longer needed.
    let rate = 0.01;
    let history = run_load_cancellation(TEST_GC_DIM, 1000, rate, 0.0);
    let first = history[0];
    let last = *history.last().unwrap();
    assert!(
        (first - 120.0).abs() < 1e-3,
        "should start carrying nothing"
    );
    assert!(
        last.abs() < 0.01 * first.abs(),
        "feedback duty only fell from {first} to {last} -- the feedforward is not taking the load"
    );
    // It must approach zero from one side rather than ringing through it.
    assert!(
        history.windows(2).all(|w| w[1] <= w[0] + 1e-3),
        "the teaching signal is not decreasing monotonically -- the step size is too large"
    );

    // The normalised granule code is supposed to make `rate` mean "fraction of the remaining error
    // corrected per step", so the error should be down by `exp(-1)` after `1 / rate` steps.
    let at_tau = history[(1.0 / rate) as usize];
    let expected = first * (-1.0f32).exp();
    assert!(
        (at_tau - expected).abs() < 0.15 * first,
        "after 1/rate steps the error was {at_tau}, expected about {expected}"
    );
}

#[test]
fn learning_speed_does_not_depend_on_how_big_the_granule_layer_is() {
    // This is the property the normalisation stage exists to provide. Without it the effective
    // step size scales with `sum_j g_j^2` and every change to --cerebellum-gc-dim or
    // --cerebellum-sparsity would silently retune how fast the arm learns.
    let rate = 0.01;
    let steps = 400;
    let small = *run_load_cancellation(1024, steps, rate, 0.0)
        .last()
        .unwrap();
    let large = *run_load_cancellation(8192, steps, rate, 0.0)
        .last()
        .unwrap();
    assert!(
        (small - large).abs() < 0.05 * 120.0,
        "an 8x larger layer learned very differently: {small} vs {large}"
    );
}

#[test]
fn the_heterosynaptic_leak_leaves_a_small_proportional_residual() {
    // The leak buys boundedness at the cost of a steady-state error: at equilibrium the Hebbian
    // and decay terms cancel at `w = cf / leak`, so some feedback duty has to remain to hold the
    // weights up. What matters in practice is that the residual is small and that it scales with
    // `leak`, so the knob behaves predictably -- pinning an exact closed form would be pinning an
    // approximation (the constant involves `sum_j g_j`, i.e. the square root of the active count).
    let load = 120.0f32;
    let small = *run_load_cancellation(TEST_GC_DIM, 6000, 0.02, 0.01)
        .last()
        .unwrap();
    let large = *run_load_cancellation(TEST_GC_DIM, 6000, 0.02, 0.1)
        .last()
        .unwrap();

    assert!(
        small < 0.01 * load,
        "even a small leak left {small} of {load} uncancelled"
    );
    assert!(
        large < 0.05 * load,
        "a 10x leak left {large} of {load} uncancelled -- more than the knob should cost"
    );
    let ratio = large / small;
    assert!(
        (5.0..20.0).contains(&ratio),
        "residual should scale roughly with leak; 10x the leak gave {ratio}x the residual"
    );
}

#[test]
fn heterosynaptic_decay_bounds_the_weights() {
    // A bare Hebbian product grows without bound once the sign of the error is consistent, which
    // is exactly what gravity produces. The leak term is what makes the rule usable; with it, the
    // fixed point is a finite weight rather than a runaway.
    let mut net = CpuNet::new(test_params());
    let mf = encode_mossy_fibres(&state_at(2000.0));
    let cf = [500.0f32; NUM_OUTPUTS]; // a large, permanently-present error
    let mut theta = 0.0f32;
    for _ in 0..20_000 {
        let (_, active) = net.step(&mf, theta, 0.0, Some(&cf), 1e-3, 1e-2);
        theta += 0.05 * (active - 0.02);
        theta = theta.clamp(-1.0, 1.0);
    }
    let peak = net.weights.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    assert!(
        peak.is_finite() && peak < 1e5,
        "weights ran away to {peak} despite the heterosynaptic decay"
    );
    // The fixed point of `cf*e - leak*w*e` is `w = cf / leak`; with cf=500 and leak=1e-2 that is
    // 50000, so anything wildly above it means the decay is not actually binding.
    assert!(
        peak <= 50_000.0 * 1.5,
        "weights overshot their fixed point: {peak}"
    );
}

#[test]
fn an_untrained_network_contributes_exactly_nothing() {
    // Enabling the cerebellum must not be able to change how the arm behaves until it has learned
    // something. That is what makes it safe to turn on with the arm already holding a position.
    let mut net = CpuNet::new(test_params());
    let mf = encode_mossy_fibres(&state_at(1200.0));
    let (ff, _) = net.step(&mf, 0.0, 0.9, None, 0.0, 0.0);
    assert_eq!(ff, [0.0; NUM_OUTPUTS]);
}

#[test]
fn plasticity_is_confined_to_joints_with_a_live_climbing_fibre() {
    // The gripper's climbing fibre is held at zero (see DEFAULT_JOINTS). Its Purkinje row must
    // therefore stay at zero even though it shares every granule cell with the joints that are
    // learning -- the parallel fibres are shared, the plasticity is not.
    let mut net = CpuNet::new(test_params());
    let mf = encode_mossy_fibres(&state_at(800.0));
    let mut cf = [80.0f32; NUM_OUTPUTS];
    cf[NUM_OUTPUTS - 1] = 0.0;
    let mut theta = 0.0f32;
    for _ in 0..500 {
        let (_, active) = net.step(&mf, theta, 0.5, Some(&cf), 1e-3, 0.0);
        theta += 0.05 * (active - 0.02);
        theta = theta.clamp(-1.0, 1.0);
    }
    let gc_dim = net.gc_dim();
    let gripper_row = &net.weights[(NUM_OUTPUTS - 1) * gc_dim..];
    assert!(
        gripper_row.iter().all(|&w| w == 0.0),
        "a joint with no climbing-fibre signal still learned"
    );
    let learned_row = &net.weights[..gc_dim];
    assert!(
        learned_row.iter().any(|&w| w != 0.0),
        "the joints that should have learned did not"
    );
}

#[test]
fn eligibility_trace_outlives_the_activity_that_set_it() {
    // The point of the trace: a climbing-fibre signal arriving after the pose has changed still
    // finds the synapses that were recently active.
    let mut net = CpuNet::new(test_params());
    let mf = encode_mossy_fibres(&state_at(600.0));
    // Prime the trace with several silent (no-plasticity) steps at this pose.
    let mut theta = 0.0f32;
    for _ in 0..200 {
        let (_, active) = net.step(&mf, theta, 0.9, None, 0.0, 0.0);
        theta += 0.05 * (active - 0.02);
        theta = theta.clamp(-1.0, 1.0);
    }
    let primed: f32 = net.trace().iter().sum();
    assert!(primed > 0.0, "trace never accumulated");

    // Silence the layer entirely; the trace must decay rather than vanish.
    for _ in 0..3 {
        net.step(&mf, 1.0, 0.9, None, 0.0, 0.0);
    }
    let after: f32 = net.trace().iter().sum();
    assert!(
        after > 0.0 && after < primed,
        "trace went from {primed} to {after} -- it should decay, not switch off"
    );
}

#[test]
fn weights_round_trip_through_a_file() {
    let dir = std::env::temp_dir().join(format!("cb_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("weights.bin");

    let weights: Vec<f32> = (0..NUM_OUTPUTS * TEST_GC_DIM)
        .map(|i| (i as f32) * 1e-4 - 3.0)
        .collect();
    save_weights(&path, &weights, TEST_GC_DIM, TEST_SEED).unwrap();
    let loaded = load_weights(&path, TEST_GC_DIM, TEST_SEED)
        .unwrap()
        .unwrap();
    assert_eq!(loaded, weights);

    // A weight vector means nothing against a different granule code, so loading must refuse
    // rather than reinterpret -- the consequence of getting this wrong is an arbitrary
    // feedforward on a real arm.
    assert!(load_weights(&path, TEST_GC_DIM, TEST_SEED ^ 1).is_err());
    assert!(load_weights(&path, TEST_GC_DIM * 2, TEST_SEED).is_err());

    // The case MF_DIM is in the header for. Widening the mossy-fibre vector reshuffles every draw
    // in `GranuleParams::generate`, but leaves gc_dim, the output count and the seed untouched --
    // so without this field the load would succeed and put an arbitrary feedforward on the arm.
    // MF_DIM is a compile-time constant, so the only way to produce that file here is to patch the
    // header directly.
    let mut wider = std::fs::read(&path).unwrap();
    wider[16..20].copy_from_slice(&((MF_DIM + 2) as u32).to_le_bytes());
    let wider_path = dir.join("wider_mf.bin");
    std::fs::write(&wider_path, &wider).unwrap();
    assert!(
        load_weights(&wider_path, TEST_GC_DIM, TEST_SEED).is_err(),
        "weights trained against {} mossy fibres were accepted for a run with {MF_DIM}",
        MF_DIM + 2
    );

    // A missing file is the normal first-run case, not an error.
    assert!(
        load_weights(&dir.join("absent.bin"), TEST_GC_DIM, TEST_SEED)
            .unwrap()
            .is_none()
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn joint_spec_parsing_rejects_out_of_range_indices() {
    assert_eq!(parse_joints("0,1,2,3,4").unwrap(), vec![0, 1, 2, 3, 4]);
    assert_eq!(parse_joints(" 1 , 3 ").unwrap(), vec![1, 3]);
    assert!(parse_joints("0,9").is_err());
    assert!(parse_joints("gripper").is_err());
    assert_eq!(parse_joints("").unwrap(), Vec::<usize>::new());
}

// ---- the live handle: threading, gating, and the safety envelope -----------------------------

use so101_impedance_ctrl::cerebellum::{
    Backend, Cerebellum, CerebellumConfig, CEREBELLUM_ACTIVE, CEREBELLUM_STALE,
};

fn handle_config() -> CerebellumConfig {
    CerebellumConfig {
        backend: Backend::Cpu,
        gc_dim: 2048,
        hz: 2000.0, // fast, so a test does not take a wall-clock minute to converge
        ..CerebellumConfig::default()
    }
}

fn now_ns() -> u64 {
    let t = std::time::SystemTime::now();
    let _ = t; // clock domain does not matter here, only that it advances monotonically
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Drives the handle the way the control loop does: publish a snapshot whose feedback duty is
/// whatever the feedforward has not yet taken over, then read back the feedforward.
fn drive(cb: &mut Cerebellum, load: f32, seconds: f32, safe: bool) -> f32 {
    let dt = 0.0025f32;
    let steps = (seconds / dt) as usize;
    let mut ff = [0f32; NUM_MOTORS];
    for _ in 0..steps {
        let mut fb_pwm = [0f32; NUM_MOTORS];
        for (out, &carried) in fb_pwm.iter_mut().zip(ff.iter()) {
            *out = load - carried;
        }
        let state = SensoryState {
            present_pos: [1500.0; NUM_MOTORS],
            fb_pwm,
            ..SensoryState::default()
        };
        cb.publish(state, safe, now_ns());
        std::thread::sleep(std::time::Duration::from_secs_f32(dt));
        let (out, _) = cb.feedforward(now_ns(), dt, safe);
        ff = out;
    }
    ff[0]
}

#[test]
fn the_live_handle_learns_a_standing_load_and_fails_safe() {
    let mut cb = Cerebellum::start(handle_config()).expect("cpu backend must always start");

    let load = 150.0f32;
    let learned = drive(&mut cb, load, 3.0, true);
    assert!(
        learned > 0.7 * load,
        "after 3 s the feedforward carried only {learned} of a {load} load"
    );

    // An unsafe tick must take the feedforward away -- a learned term cannot be the one part of
    // the controller that survives its own fail-safe.
    let after_unsafe = drive(&mut cb, load, 1.0, false);
    assert!(
        after_unsafe.abs() < 1.0,
        "feedforward was still {after_unsafe} on unsafe ticks"
    );

    cb.stop();
}

#[test]
fn the_feedforward_is_slew_limited_in_both_directions() {
    let mut cfg = handle_config();
    cfg.ff_slew = 100.0; // duty per second
    let mut cb = Cerebellum::start(cfg).expect("cpu backend must always start");

    // Learn something worth dropping.
    drive(&mut cb, 200.0, 3.0, true);

    // One tick of fail-safe must not step the command to zero; it has to ramp.
    let dt = 0.01f32;
    let (before, _) = cb.feedforward(now_ns(), 0.0, true);
    let (after, _) = cb.feedforward(now_ns(), dt, false);
    let step = (after[0] - before[0]).abs();
    assert!(
        step <= 100.0 * dt + 1e-3,
        "feedforward moved {step} in one {dt}s tick, above the {} /s limit",
        100.0
    );

    cb.stop();
}

#[test]
fn a_silent_cerebellum_goes_stale_rather_than_holding_its_last_output() {
    let mut cfg = handle_config();
    cfg.staleness_ms = 20;
    cfg.hz = 1.0; // publishes once a second, so it is stale almost immediately
    let mut cb = Cerebellum::start(cfg).expect("cpu backend must always start");

    std::thread::sleep(std::time::Duration::from_millis(200));
    let (ff, flags) = cb.feedforward(now_ns(), 0.01, true);
    assert_eq!(ff, [0.0; NUM_MOTORS]);
    assert!(flags & CEREBELLUM_STALE != 0, "stale output not flagged");
    assert!(
        flags & CEREBELLUM_ACTIVE == 0,
        "stale output reported active"
    );

    cb.stop();
}

#[test]
fn the_gripper_never_receives_a_feedforward() {
    // DEFAULT_JOINTS leaves the gripper out, and that has to hold all the way through to the duty
    // the control loop applies -- a gripper that learns its own grasp keeps squeezing.
    let mut cb = Cerebellum::start(handle_config()).expect("cpu backend must always start");
    drive(&mut cb, 200.0, 2.0, true);
    let (ff, _) = cb.feedforward(now_ns(), 0.0025, true);
    assert_eq!(
        ff[NUM_MOTORS - 1],
        0.0,
        "the gripper was given a feedforward of {}",
        ff[NUM_MOTORS - 1]
    );
    assert!(
        ff[0] != 0.0,
        "the arm joints learned nothing, so this proves little"
    );
    cb.stop();
}

/// An ordinary snapshot is believable, including one working hard.
///
/// The bounds have to leave the arm's whole real operating range alone, or the cerebellum stops
/// stepping exactly when it is most useful.
#[test]
fn a_normal_sensory_snapshot_is_believable() {
    let mut state = state_at(1200.0);
    state.present_vel = [1650.0; NUM_MOTORS]; // the fastest hand-driven motion measured
    state.pos_error = [-19.0; NUM_MOTORS];
    state.present_current = [-69.0; NUM_MOTORS]; // the largest holding current measured
    assert_eq!(implausible_input(&state), None);
}

/// The measured corruption: `Present_Current` read as a raw sign-magnitude word.
///
/// This is the reading that put six of the thirty mossy fibres on the rail at once and drove the
/// feedforward for an unloaded joint to its clamp. It is 164 encoding scales out; the bound is 20.
#[test]
fn the_measured_current_corruption_is_refused() {
    let mut state = state_at(1200.0);
    state.present_current[3] = 32790.0;
    let (channel, joint, value) = implausible_input(&state).expect("should be refused");
    assert_eq!(channel, "current");
    assert_eq!(joint, 3);
    assert_eq!(value, 32790.0);
}

/// A position outside the encoder is not a pose. Unlike the squashed channels this one has a hard
/// bound rather than a derived one -- the reading is defined modulo 4096, so anything outside it
/// came from somewhere other than the encoder.
#[test]
fn a_position_outside_the_encoder_is_refused() {
    let mut state = state_at(1200.0);
    state.present_pos[1] = 5000.0;
    let (channel, joint, _) = implausible_input(&state).expect("should be refused");
    assert_eq!((channel, joint), ("position", 1));

    let mut nan = state_at(1200.0);
    nan.present_pos[0] = f32::NAN;
    assert!(
        implausible_input(&nan).is_some(),
        "NaN is not a pose either"
    );
}

/// The bound sits past where the encoding stops carrying information, so refusing a value cannot
/// discard anything the network could have used.
///
/// `tanh` reaches 1.0 in `f32` by about ten scales; the factor is twice that. Both of these encode
/// to exactly the same mossy fibre, and only one of them is refused -- which is the point: by then
/// the input is indistinguishable, so the only question left is whether it is plausible.
#[test]
fn the_bound_sits_where_the_encoding_has_already_saturated() {
    let mut at_bound = state_at(1200.0);
    at_bound.present_current[0] = 200.0 * MF_SATURATION_FACTOR;
    let mut past_bound = state_at(1200.0);
    past_bound.present_current[0] = 200.0 * MF_SATURATION_FACTOR * 2.0;

    let a = encode_mossy_fibres(&at_bound);
    let b = encode_mossy_fibres(&past_bound);
    assert_eq!(a[4], b[4], "both saturate to the same mossy fibre");
    assert_eq!(a[4], 1.0);

    assert_eq!(implausible_input(&at_bound), None);
    assert!(implausible_input(&past_bound).is_some());
}
