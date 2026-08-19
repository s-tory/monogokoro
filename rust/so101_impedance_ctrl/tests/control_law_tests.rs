//! Pure control-law math tests -- clamping, moving average, watchdog behavior. No hardware or
//! shared memory involved.

use so101_impedance_ctrl::control::{
    finite_difference_velocity, impedance_pwm, input_is_fresh, MovingAverage,
};

#[test]
fn impedance_law_computes_spring_damper_sum() {
    // K*(target-present) + D*(target_vel-present_vel) = 10*(1.0-0.5) + 2*(0.5-0.4) = 5.2
    let pwm = impedance_pwm(10.0, 2.0, 1.0, 0.5, 0.5, 0.4, 1000.0);
    assert!((pwm - 5.2).abs() < 1e-4, "got {pwm}");
}

#[test]
fn impedance_law_clamps_to_pwm_max_both_directions() {
    let pwm = impedance_pwm(1000.0, 0.0, 10.0, 0.0, 0.0, 0.0, 500.0);
    assert_eq!(pwm, 500.0);

    let pwm_neg = impedance_pwm(1000.0, 0.0, -10.0, 0.0, 0.0, 0.0, 500.0);
    assert_eq!(pwm_neg, -500.0);
}

#[test]
fn impedance_law_zero_gains_produce_zero_pwm() {
    let pwm = impedance_pwm(0.0, 0.0, 10.0, -10.0, 5.0, -5.0, 1000.0);
    assert_eq!(pwm, 0.0);
}

#[test]
fn moving_average_tracks_a_fixed_window() {
    let mut avg = MovingAverage::new(3);
    assert_eq!(avg.push(10.0), 10.0);
    assert_eq!(avg.push(20.0), 15.0);
    assert_eq!(avg.push(30.0), 20.0);
    // Window is now full: pushing a 4th sample drops the oldest (10.0).
    assert_eq!(avg.push(60.0), (20.0 + 30.0 + 60.0) / 3.0);
}

#[test]
fn watchdog_flags_stale_input_but_passes_fresh_input() {
    assert!(input_is_fresh(1_000_000_000, 950_000_000, 75_000_000));
    assert!(!input_is_fresh(1_000_000_000, 900_000_000, 75_000_000));
    // Exactly at the boundary counts as fresh.
    assert!(input_is_fresh(1_075_000_000, 1_000_000_000, 75_000_000));
}

#[test]
fn finite_difference_velocity_matches_slope() {
    let v = finite_difference_velocity(1.0, 1.5, 0.1);
    assert!((v - 5.0).abs() < 1e-4);
}

#[test]
fn finite_difference_velocity_guards_against_nonpositive_dt() {
    assert_eq!(finite_difference_velocity(1.0, 2.0, 0.0), 0.0);
    assert_eq!(finite_difference_velocity(1.0, 2.0, -0.1), 0.0);
}

#[test]
fn wrapped_delta_takes_the_short_way_round_the_seam() {
    use so101_impedance_ctrl::control::wrapped_delta;
    // Straight subtraction would call this -4090; it is really +6 counts across the 4095/0 seam.
    assert!((wrapped_delta(3.0, 4093.0) - 6.0).abs() < 1e-3);
    assert!((wrapped_delta(4093.0, 3.0) + 6.0).abs() < 1e-3);
    // Well away from the seam it must behave like plain subtraction.
    assert!((wrapped_delta(2000.0, 1900.0) - 100.0).abs() < 1e-3);
    // Exactly half a revolution is the boundary; magnitude never exceeds 2048.
    assert!(wrapped_delta(4000.0, 1000.0).abs() <= 2048.0);
}

#[test]
fn impedance_law_does_not_slam_across_the_encoder_seam() {
    use so101_impedance_ctrl::control::impedance_pwm;
    // Target 3 counts past the seam from present 4093: a 6-count error, so a gentle command --
    // not the saturated one a naive `target - present` would produce.
    let pwm = impedance_pwm(10.0, 0.0, 3.0, 4093.0, 0.0, 0.0, 1000.0);
    assert!((pwm - 60.0).abs() < 1e-3, "got {pwm}");
}

#[test]
fn velocity_estimate_ignores_a_rollover_between_samples() {
    // One count of real motion across the seam, at 1 kHz: 1000 counts/s, not ~4 million.
    let v = finite_difference_velocity(4095.0, 0.0, 0.001);
    assert!((v - 1000.0).abs() < 1.0, "got {v}");
}
