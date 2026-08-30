//! Holds the compute shaders against the CPU reference they were written from.
//!
//! A compute shader is not debuggable by reading it, and its failure mode is not a crash -- it is
//! plausible-looking wrong numbers, which downstream become a confidently wrong feedforward on a
//! real arm. So the readable implementation in `net.rs` is the authority and this checks that the
//! fast one agrees with it, step by step, through learning.
//!
//! Every test here skips (rather than fails) when no Vulkan device is available, so the suite
//! still runs on a headless CI machine. On the robot host it must actually run: a skipped
//! cross-check is not a passing one.

use std::time::Instant;

use so101_impedance_ctrl::cerebellum::gpu::GpuNet;
use so101_impedance_ctrl::cerebellum::net::{
    encode_mossy_fibres, CpuNet, GranuleParams, SensoryState, NUM_OUTPUTS,
};
use so101_impedance_ctrl::shm::NUM_MOTORS;

const GC_DIM: usize = 4096;
const SEED: u64 = 0x0C0F_FEE0_1234_5678;

/// Builds the GPU backend, or prints why it could not and returns `None`.
fn gpu_or_skip(params: &GranuleParams) -> Option<GpuNet> {
    match GpuNet::new(params, true) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP: no usable Vulkan device ({e})");
            None
        }
    }
}

fn state_at(t: f32) -> SensoryState {
    let mut s = SensoryState::default();
    for j in 0..NUM_MOTORS {
        s.present_pos[j] = (t * 13.0 + j as f32 * 611.0) % 4096.0;
        s.present_vel[j] = (t * 0.7).sin() * 40.0;
        s.pos_error[j] = (t * 0.3 + j as f32).cos() * 25.0;
        s.present_current[j] = 60.0 + j as f32 * 5.0;
    }
    s
}

#[test]
fn shaders_agree_with_the_cpu_reference_through_learning() {
    let params = GranuleParams::generate(GC_DIM, SEED);
    let Some(mut gpu) = gpu_or_skip(&params) else {
        return;
    };
    let mut cpu = CpuNet::new(GranuleParams::generate(GC_DIM, SEED));

    let mut theta = 0.0f32;
    let rate = 0.01;
    let leak = 0.05;
    let trace_decay = 0.9;

    for step in 0..300 {
        let t = step as f32;
        let mf = encode_mossy_fibres(&state_at(t));
        // A varying teaching signal, so the eligibility trace and the weight update are both
        // genuinely exercised rather than settling into a fixed point that hides a discrepancy.
        let mut cf = [0f32; NUM_OUTPUTS];
        for (i, c) in cf.iter_mut().enumerate() {
            *c = 50.0 * (t * 0.05 + i as f32).sin();
        }
        // Output 0's climbing fibre falls silent halfway through -- what the deadband produces
        // once a joint's feedforward has taken over its load, and what a gated-out joint looks
        // like all run. Both sides have to freeze that row instead of decaying it, and an exact
        // zero is the only value that reaches the branch which does so.
        if step >= 150 {
            cf[0] = 0.0;
        }

        let (gpu_ff, gpu_active) = gpu
            .step(&mf, theta, trace_decay, Some(&cf), rate, leak)
            .expect("GPU step failed");
        let (cpu_ff, cpu_active) = cpu.step(&mf, theta, trace_decay, Some(&cf), rate, leak);

        // The active count is an integer both sides compute the same way, so this one must match
        // exactly -- a mismatch means the Golgi threshold is being applied differently.
        assert_eq!(
            gpu_active, cpu_active,
            "step {step}: granule sparsity differs ({gpu_active} vs {cpu_active})"
        );

        for i in 0..NUM_OUTPUTS {
            let (a, b) = (gpu_ff[i], cpu_ff[i]);
            // The shader's tree reduction sums in a different order from the reference's
            // sequential loop, so they agree to floating-point tolerance rather than bit-exactly.
            let tol = 1e-3 * b.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "step {step}, output {i}: GPU {a} vs CPU {b}"
            );
        }

        theta += 0.05 * (gpu_active - 0.02);
        theta = theta.clamp(-1.0, 1.0);
    }

    // Divergence would most likely accumulate in the weights rather than show up in one step's
    // readout, so check the learned state itself at the end.
    let gpu_w = gpu.read_weights();
    let cpu_w = &cpu.weights;
    assert_eq!(gpu_w.len(), cpu_w.len());
    let worst = gpu_w
        .iter()
        .zip(cpu_w.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = cpu_w.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    assert!(
        worst <= 1e-3 * scale.max(1.0),
        "learned weights drifted apart: worst |delta| = {worst} against a peak weight of {scale}"
    );
}

#[test]
fn an_untrained_gpu_network_outputs_exactly_zero() {
    // Same guarantee the CPU path gives: enabling the cerebellum cannot change how the arm behaves
    // until it has learned something.
    let params = GranuleParams::generate(GC_DIM, SEED);
    let Some(mut gpu) = gpu_or_skip(&params) else {
        return;
    };
    let mf = encode_mossy_fibres(&state_at(3.0));
    let (ff, active) = gpu.step(&mf, 0.0, 0.9, None, 0.0, 0.0).unwrap();
    assert_eq!(ff, [0.0; NUM_OUTPUTS]);
    assert!(active > 0.0, "the granule layer produced nothing at all");
}

#[test]
fn gpu_weights_survive_a_round_trip_through_the_host() {
    let params = GranuleParams::generate(GC_DIM, SEED);
    let Some(mut gpu) = gpu_or_skip(&params) else {
        return;
    };
    let weights: Vec<f32> = (0..NUM_OUTPUTS * GC_DIM)
        .map(|i| (i % 97) as f32 * 0.01 - 0.5)
        .collect();
    gpu.write_weights(&weights).unwrap();
    assert_eq!(gpu.read_weights(), weights);

    // A wrongly-sized vector must be refused, not partially applied.
    assert!(gpu.write_weights(&weights[..10]).is_err());
}

#[test]
fn report_per_step_latency() {
    // Not an assertion -- a measurement, printed so the number in the README stays honest. Run
    // with `--nocapture` to see it.
    let mf = encode_mossy_fibres(&state_at(1.0));
    let cf = [10.0f32; NUM_OUTPUTS];

    // Swept, because the interesting result is the *shape*: if cost barely moves with layer size
    // then the step is submission latency rather than compute, and buying a bigger granule layer
    // is nearly free.
    for gc_dim in [1024usize, 4096, 16384, 65536] {
        let params = GranuleParams::generate(gc_dim, SEED);
        let Some(mut gpu) = gpu_or_skip(&params) else {
            return;
        };
        for _ in 0..50 {
            gpu.step(&mf, 0.0, 0.9, Some(&cf), 0.01, 0.05).unwrap();
        }
        let n = 500;
        let mut worst = 0u128;
        let start = Instant::now();
        for _ in 0..n {
            let t = Instant::now();
            gpu.step(&mf, 0.0, 0.9, Some(&cf), 0.01, 0.05).unwrap();
            worst = worst.max(t.elapsed().as_micros());
        }
        let mean = start.elapsed().as_micros() / n as u128;
        eprintln!(
            "cerebellum GPU step, gc_dim={gc_dim:>6}: mean {mean:>4} us, max {worst:>4} us \
             (submit + fence wait, not shader time)"
        );
    }
}
