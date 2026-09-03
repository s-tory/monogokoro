//! Pure control-law math tests -- clamping, moving average, watchdog behavior. No hardware or
//! shared memory involved.

use so101_impedance_ctrl::control::{
    apply_soft_limits, finite_difference_velocity, first_implausible_step, first_outside_travel,
    impedance_pwm, input_is_fresh, MovingAverage, PositionFrame, PositionGate, TravelEnvelope,
    TravelVerdict,
};

/// One tick's worth of budget at the shipped defaults: 20000 counts/s at 400 Hz.
const TICK_BUDGET: f32 = 50.0;

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

#[test]
fn averaging_finite_differences_equals_the_wide_window_difference() {
    use so101_impedance_ctrl::control::MovingAverage;

    // The velocity filter is a moving average over per-tick differences rather than a difference
    // against a saved older sample. Those are the same quantity -- the intermediate terms
    // telescope -- which is what lets the filter divide quantisation noise by the window size
    // while still reporting a true velocity rather than a scaled-down one.
    let n = 8;
    let dt = 1.0 / 400.0;
    let mut avg = MovingAverage::new(n);

    // A quantised ramp: the encoder ticks over on some samples and not others, exactly the
    // pattern that makes the raw single-tick difference alternate between 0 and 400 ticks/s.
    let positions: Vec<f32> = (0..=n).map(|i| 1000.0 + (i as f32 * 0.4).floor()).collect();
    let mut filtered = 0.0;
    for w in positions.windows(2) {
        filtered = avg.push(finite_difference_velocity(w[0], w[1], dt));
    }

    let wide = (positions[n] - positions[0]) / (n as f32 * dt);
    assert!(
        (filtered - wide).abs() < 1e-2,
        "filtered {filtered} vs wide {wide}"
    );
}

#[test]
fn velocity_filter_preserves_a_steady_velocity() {
    use so101_impedance_ctrl::control::MovingAverage;

    // Filtering must not attenuate the signal, only the noise: a joint moving at a constant rate
    // has to report that rate once the window has filled, or the D term would be silently scaled.
    let dt = 1.0 / 400.0;
    let mut avg = MovingAverage::new(8);
    let mut filtered = 0.0;
    let mut pos = 1000.0f32;
    for _ in 0..32 {
        let next = pos + 2.0; // 2 counts/tick == 800 counts/s
        filtered = avg.push(finite_difference_velocity(pos, next, dt));
        pos = next;
    }
    assert!((filtered - 800.0).abs() < 1e-2, "got {filtered}");
}

#[test]
fn force_feedback_is_slack_until_the_follower_is_actually_blocked() {
    use so101_impedance_ctrl::control::force_feedback_pwm;
    // A gripper closing freely tracks its target, so the operator should feel nothing. Any
    // standing resistance with no object present would be felt as a stiff, lying trigger.
    let pwm = force_feedback_pwm(2.0, 0.2, 0.0, 0.0, 250.0);
    assert!((pwm - 0.0).abs() < 1e-6, "got {pwm}");
}

#[test]
fn force_feedback_scales_with_the_followers_tracking_error() {
    use so101_impedance_ctrl::control::force_feedback_pwm;
    // Squeezing harder against a blocked gripper widens target-vs-present, which is the whole
    // signal: the operator feels the object push back harder the harder they ask for.
    let light = force_feedback_pwm(2.0, 0.0, 10.0, 0.0, 250.0);
    let hard = force_feedback_pwm(2.0, 0.0, 40.0, 0.0, 250.0);
    assert!((light - 20.0).abs() < 1e-3, "got {light}");
    assert!((hard - 80.0).abs() < 1e-3, "got {hard}");
}

#[test]
fn force_feedback_damping_opposes_motion_whichever_way_the_gain_points() {
    use so101_impedance_ctrl::control::force_feedback_pwm;
    // The correct sign of the gain depends on the two grippers' calibrations, but damping must
    // resist the operator's hand regardless -- otherwise negating the gain would turn the damper
    // into an accelerator.
    for gain in [2.0_f32, -2.0_f32] {
        let opening = force_feedback_pwm(gain, 0.5, 0.0, 100.0, 250.0);
        let closing = force_feedback_pwm(gain, 0.5, 0.0, -100.0, 250.0);
        assert!(opening < 0.0, "gain {gain}: got {opening}");
        assert!(closing > 0.0, "gain {gain}: got {closing}");
    }
}

#[test]
fn force_feedback_clamps_to_the_leaders_own_lower_limit() {
    use so101_impedance_ctrl::control::force_feedback_pwm;
    // This bound is what a human hand is holding, not a gearbox: a sign-flipped or over-large gain
    // must saturate somewhere the operator can still overpower.
    let pwm = force_feedback_pwm(50.0, 0.0, 500.0, 0.0, 250.0);
    assert!((pwm - 250.0).abs() < 1e-3, "got {pwm}");
    let pwm = force_feedback_pwm(-50.0, 0.0, 500.0, 0.0, 250.0);
    assert!((pwm + 250.0).abs() < 1e-3, "got {pwm}");
}

/// Ordinary motion is not a glitch. The servo's own no-load speed is ~3100 counts/s, which is
/// ~8 counts per 2.5 ms tick -- well inside one tick's budget.
#[test]
fn plausible_motion_is_accepted() {
    let prev = [400.0, 900.0, 3400.0, 170.0, 2900.0, 3900.0];
    let values = [408, 892, 3406, 170, 2900, 3900];
    assert_eq!(first_implausible_step(&values, &prev, TICK_BUDGET), None);
}

/// The sample that ran the arm into its encoder wrap: `wrist_flex` reported 3784 counts of travel
/// in one 2.5 ms tick. Nothing raised, nothing timed out -- the reflex simply answered it.
#[test]
fn the_measured_misparse_is_rejected_and_names_its_motor() {
    let prev = [400.0, 900.0, 3400.0, 121.0, 2900.0, 3900.0];
    let values = [400, 900, 3400, 3905, 2900, 3900];
    let (i, step) = first_implausible_step(&values, &prev, TICK_BUDGET).expect("rejected");
    assert_eq!(i, 3);
    // Wrapped, the shorter way round: the joint apparently went 312 counts *down* through zero,
    // not 3784 counts up. Either way it is ~125,000 counts/s, and neither is a thing an SO-101
    // joint does.
    assert!((step + 312.0).abs() < 1e-3, "got {step}");
}

/// A joint crossing 4095/0 moves one count while its raw reading moves 4095. Comparing raw would
/// make the check fire hardest exactly where the encoder is most awkward.
#[test]
fn crossing_the_encoder_wrap_is_not_a_glitch() {
    let prev = [400.0, 900.0, 3400.0, 4090.0, 2900.0, 3900.0];
    let values = [400, 900, 3400, 5, 2900, 3900];
    assert_eq!(first_implausible_step(&values, &prev, TICK_BUDGET), None);
}

/// A real excursion during a blind run gets in, but only once a second read agrees with it.
///
/// Measured case: the bus went quiet for ~1 s, the fail-safe zeroed PWM, and the gravity-loaded
/// arm fell ~500 counts before telemetry returned.
#[test]
fn a_fall_during_a_blind_run_is_admitted_once_a_second_read_agrees() {
    let prev = [400.0, 961.0, 3400.0, 170.0, 2900.0, 3900.0];
    let fallen = [400, 1505, 3400, 170, 2900, 3900];
    let mut gate = PositionGate::new(TICK_BUDGET);
    gate.accept(&fallen, None)
        .expect("the first batch seeds the gate");

    // Now with a history: one tick cannot cover 544 counts, so the batch is held, not used.
    assert!(gate.accept(&fallen, Some(&prev)).is_err());
    // The next read says the same thing, so the arm really is down there.
    assert!(gate.accept(&fallen, Some(&prev)).is_ok());
}

/// The measured failure of the first version: a misparse that arrives *after* a blackout.
///
/// `wrist_flex` was really at 1187 and one reply put it at 0. The gate that widened with the
/// length of the blind run accepted it, because by then it had stopped checking. Corroboration
/// does not, because the next read disagrees -- which is what a misparse looks like and what a
/// real position does not.
#[test]
fn a_misparse_after_a_blackout_is_still_refused() {
    let prev = [400.0, 961.0, 3400.0, 1187.0, 2900.0, 3900.0];
    let garbage = [400, 961, 3400, 0, 2900, 3900];
    let truth = [400, 961, 3400, 1187, 2900, 3900];
    let mut gate = PositionGate::new(TICK_BUDGET);
    gate.accept(&truth, None).expect("seed");

    let first = gate
        .accept(&garbage, Some(&prev))
        .expect_err("held, not used");
    assert_eq!(first.motor, 3);
    assert!(!first.corroborating, "nothing was being held yet");

    // The real position comes back next tick. It must not be mistaken for corroboration of the
    // garbage, and it agrees with `prev`, so it is simply accepted.
    assert!(gate.accept(&truth, Some(&prev)).is_ok());
}

/// Two different wrong readings in a row do not corroborate each other.
#[test]
fn corroboration_needs_the_same_answer_twice() {
    let prev = [400.0, 961.0, 3400.0, 1187.0, 2900.0, 3900.0];
    let mut gate = PositionGate::new(TICK_BUDGET);
    gate.accept(&[400, 961, 3400, 1187, 2900, 3900], None)
        .expect("seed");

    let a = gate
        .accept(&[400, 961, 3400, 0, 2900, 3900], Some(&prev))
        .expect_err("held");
    assert!(!a.corroborating);
    let b = gate
        .accept(&[400, 961, 3400, 2500, 2900, 3900], Some(&prev))
        .expect_err("a different wrong answer is not agreement");
    assert!(
        b.corroborating,
        "the gate was holding one when this arrived"
    );
}

/// A batch is condemned by its first bad slot, whichever motor that is -- a reply whose framing
/// slipped shifts every slot after it, so the rest of the batch is not evidence of anything.
#[test]
fn the_first_implausible_motor_is_the_one_reported() {
    let prev = [400.0, 900.0, 3400.0, 121.0, 2900.0, 3900.0];
    let values = [400, 2000, 3400, 3905, 2900, 3900];
    let (i, _) = first_implausible_step(&values, &prev, TICK_BUDGET).expect("rejected");
    assert_eq!(i, 1);
}

/// Inside the limits the function is a pass-through, whatever it remembers.
#[test]
fn soft_limits_do_not_touch_a_joint_in_range() {
    assert_eq!(
        apply_soft_limits(500.0, 2000.0, 100.0, 3995.0, Some(2000.0)),
        500.0
    );
    assert_eq!(
        apply_soft_limits(-500.0, 2000.0, 100.0, 3995.0, None),
        -500.0
    );
}

/// Past a limit with no seam in the way, the remembered position and the raw rule agree: outward
/// is blocked, inward is allowed.
#[test]
fn past_a_limit_only_the_way_back_is_allowed() {
    // Past pos_max, last seen at 3000 -- back is negative.
    assert_eq!(
        apply_soft_limits(500.0, 4000.0, 100.0, 3995.0, Some(3000.0)),
        0.0
    );
    assert_eq!(
        apply_soft_limits(-500.0, 4000.0, 100.0, 3995.0, Some(3000.0)),
        -500.0
    );
}

/// The measured trap. A joint at 4051 whose target is 123 is *past `pos_max`* by the raw rule, so
/// the positive command that is the short way home looks like driving further out. Having watched
/// it leave from 121 is what turns that around.
#[test]
fn a_joint_that_left_through_the_seam_is_driven_back_through_it() {
    let (pwm, pos) = (500.0, 4051.0);
    // The raw rule -- what `None` falls back to -- blocks the only useful direction.
    assert_eq!(apply_soft_limits(pwm, pos, 100.0, 3995.0, None), 0.0);
    // Remembering 121 makes the short way round (+166) the escape, so the same command is allowed.
    assert_eq!(apply_soft_limits(pwm, pos, 100.0, 3995.0, Some(121.0)), pwm);
    // ...and the other direction, which would drive it deeper past pos_max, is not.
    assert_eq!(
        apply_soft_limits(-pwm, pos, 100.0, 3995.0, Some(121.0)),
        0.0
    );
}

/// A daemon started with the arm already outside its limits has no observation to appeal to. It
/// must not invent one: the raw rule still keeps it off the seam, and the fault flag asks a human.
#[test]
fn without_history_the_raw_rule_still_applies() {
    assert_eq!(apply_soft_limits(500.0, 4051.0, 100.0, 3995.0, None), 0.0);
    assert_eq!(
        apply_soft_limits(-500.0, 4051.0, 100.0, 3995.0, None),
        -500.0
    );
    assert_eq!(apply_soft_limits(-500.0, 50.0, 100.0, 3995.0, None), 0.0);
    assert_eq!(apply_soft_limits(500.0, 50.0, 100.0, 3995.0, None), 500.0);
}

/// This arm's calibration as the servos hold it, widened by the shipped 200-count margin.
///
/// Travel measured 2026-09-02; `Homing_Offset` read off the same six servos 2026-09-03. Two of the
/// offsets are negative, which is why the fold is modulo a revolution rather than an add.
///
/// `wrist_roll` has no envelope, and the reason changed: it is not that the joint spins. Measured
/// 2026-09-03 it reaches 113-3981 and is stopped both ways by a printed corner -- but its
/// registers still read `0-4095`, because the calibration that wrote them never swept it. Until
/// this arm is recalibrated, a full-width travel and an unmeasured one are the same thing here.
fn measured_envelopes() -> [Option<TravelEnvelope>; 6] {
    let band = |lo: f32, hi: f32, offset: f32| TravelEnvelope::new(lo, hi, offset, 200.0);
    [
        band(859.0, 3285.0, 1739.0),   // shoulder_pan
        band(946.0, 3343.0, -1242.0),  // shoulder_lift
        band(783.0, 2940.0, 1484.0),   // elbow_flex
        band(872.0, 3263.0, -1917.0),  // wrist_flex
        band(0.0, 4095.0, 1932.0),     // wrist_roll: 0-4095, no constraint
        band(1992.0, 3566.0, 1867.0),  // gripper
    ]
}

/// The frame each joint reports in while the arm is limp on the bench, and while it is driven.
const ALL_CORRECTED: [Option<PositionFrame>; 6] = [Some(PositionFrame::Corrected); 6];
const ALL_RAW: [Option<PositionFrame>; 6] = [Some(PositionFrame::Raw); 6];

/// A joint calibrated over so much of the circle that the margin laps it gets no envelope.
///
/// 3900 counts of travel widened by 200 either side is 4300 counts of arc on a 4096-count circle:
/// every position is inside it. A check that admits everything is worse than no check, because it
/// reads like protection.
#[test]
fn travel_the_margin_laps_the_circle_gets_no_envelope() {
    assert!(TravelEnvelope::new(0.0, 3900.0, 0.0, 200.0).is_none());
    assert!(TravelEnvelope::new(0.0, 3600.0, 0.0, 200.0).is_none());
    assert!(TravelEnvelope::new(0.0, 3599.0, 0.0, 200.0).is_some());
    // A travel that is empty or inverted is a calibration nobody can enforce either.
    assert!(TravelEnvelope::new(2000.0, 2000.0, 0.0, 200.0).is_none());
    assert!(TravelEnvelope::new(3000.0, 900.0, 0.0, 200.0).is_none());
}

/// A pose from the middle of the arm's travel is not a glitch, envelope or no envelope.
#[test]
fn positions_inside_the_travel_are_accepted() {
    let values = [1995, 2294, 2158, 2153, 2003, 2048];
    assert_eq!(
        first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED),
        None
    );
}

/// The extremes of a real hand sweep, measured the same day the envelopes were: every joint
/// pressed to its stop still reads inside, which is what the margin is for.
#[test]
fn the_measured_hand_sweep_stays_inside_the_envelope() {
    for values in [
        [854, 944, 738, 888, 105, 1997],   // low ends, including elbow 45 counts under its travel
        [3294, 3350, 2942, 3260, 3955, 3579], // high ends
    ] {
        assert_eq!(
            first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED),
            None,
            "{values:?}"
        );
    }
}

/// **The bug this whole check was turned off for.** Every joint, in PWM, reading exactly what the
/// servos reported on 2026-09-03 -- the state the daemon is in for an entire driven run.
///
/// The first version of the gate compared these against the position-mode envelope and rejected
/// all six, which counts blind ticks and ends in `--max-blind-ticks` zeroing PWM on a loaded arm.
/// It was not caught on the bench because a limp arm is in position mode, the one state where the
/// two frames agree.
#[test]
fn the_measured_pwm_frame_readings_are_accepted() {
    let values = [3814, 3814, 325, 517, 3976, 4068];
    assert_eq!(
        first_outside_travel(&values, &measured_envelopes(), &ALL_RAW),
        None
    );
}

/// The same six readings, told the arm is in position mode: refused, and the daemon says why.
///
/// This is the shape of the old failure preserved as a test. What makes it diagnosable rather than
/// mysterious is that *all* the checked joints land inside the other frame at once, which no bad
/// read does -- the client switches all six together, so only a missed mode switch looks like this.
#[test]
fn a_missed_mode_switch_is_refused_and_named() {
    let values = [3814, 3814, 325, 517, 3976, 4068];
    let reject = first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED)
        .expect("PWM-frame readings cannot be inside the position-mode envelopes");
    assert_eq!(
        reject.verdict,
        TravelVerdict::OutsideTravel {
            every_checked_joint_fits_the_other_frame: true
        }
    );
    assert!(reject.reason().contains("mode switch this daemon did not see"));
}

/// One joint outside its travel is a bad read, not a frame error, and must not be reported as one.
///
/// On this arm a joint's two envelopes cover most of the circle between them, so a single wrong
/// value lands inside the other frame's band about as often as not. The claim is only made when
/// every checked joint makes it at once.
#[test]
fn one_joint_outside_is_not_blamed_on_the_frame() {
    let values = [4083, 2294, 2158, 2153, 2003, 2048];
    let reject = first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED).unwrap();
    assert_eq!(reject.motor, 0);
    assert_eq!(
        reject.verdict,
        TravelVerdict::OutsideTravel {
            every_checked_joint_fits_the_other_frame: false
        }
    );
}

/// The fault this exists for. `shoulder_pan` reported whole-turn jumps three times in one 25-second
/// sweep while its hard stops sat at 867 and 3280, repeatable to 3-6 counts, with 815 counts of
/// clearance to the wrap. Every one of those readings is outside the travel and none of them could
/// have been where the joint was. Measured with the arm limp, so: position mode.
#[test]
fn the_measured_whole_turn_misreport_is_rejected() {
    for (reported, label) in [(0, "0"), (3, "3"), (10, "10"), (4083, "4083"), (4094, "4094")] {
        let values = [reported, 2294, 2158, 2153, 2003, 2048];
        let reject = first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED)
            .expect(label);
        assert_eq!(reject.motor, 0, "{label}");
        assert_eq!(reject.value, reported as f32, "{label}");
    }
}

/// The same joint while driven, i.e. in the raw frame, where its band is 2398-1128 and the arc it
/// cannot be in is 1129-2397.
///
/// The fold moves *which* readings are impossible, and this is what that costs: the whole-turn
/// misreports measured in position mode (0, 3, 10, 4094) are all perfectly ordinary raw positions
/// for this joint, so in a driven run they would pass. 1269 counts of the circle stay impossible,
/// and that is what is still being checked -- not the same 1269 counts, and worth knowing before
/// reading a clean driven run as evidence the gate caught anything.
#[test]
fn the_whole_turn_misreport_is_still_rejected_in_the_raw_frame() {
    for (reported, expected) in [
        (0, None),
        (3, None),
        (10, None),
        (4094, None),
        (1000, None),
        (1129, Some(1129.0)),
        (2000, Some(2000.0)),
        (2397, Some(2397.0)),
        (2398, None),
        (2500, None),
    ] {
        let values = [reported, 3814, 325, 517, 3976, 4068];
        let got = first_outside_travel(&values, &measured_envelopes(), &ALL_RAW)
            .map(|r| r.value);
        assert_eq!(got, expected, "shoulder_pan reporting {reported} in the raw frame");
    }
}

/// A position the encoder cannot report is refused whether or not the joint has an envelope, and
/// is named as a misparsed reply rather than blamed on the calibration.
///
/// The envelope is folded modulo a revolution and the value is deliberately not: folding
/// `2074 + 4096` would tuck a whole-turn misreport neatly back inside the arc and hide exactly the
/// fault this gate was written for.
#[test]
fn a_position_off_the_encoder_is_refused_without_a_calibration() {
    let no_envelopes = [None, None, None, None, None, None];
    for reported in [6170, 4096, 65535, -1] {
        for envelopes in [measured_envelopes(), no_envelopes] {
            let values = [reported, 2294, 2158, 2153, 2003, 2048];
            let reject = first_outside_travel(&values, &envelopes, &ALL_CORRECTED)
                .unwrap_or_else(|| panic!("{reported} was accepted"));
            assert_eq!(reject.motor, 0);
            assert_eq!(reject.verdict, TravelVerdict::OffEncoder);
        }
    }
}

/// The envelope is not corroborable, and that is the point: the slew gate would let a steady wrong
/// reading through on the second look, because agreeing with itself is all corroboration asks for.
#[test]
fn a_steady_misreport_corroborates_itself_past_the_slew_gate() {
    let prev = [2477.0, 2294.0, 2158.0, 2153.0, 2003.0, 2048.0];
    let values = [4083, 2294, 2158, 2153, 2003, 2048];
    let mut gate = PositionGate::new(TICK_BUDGET);

    // First look: too far in one tick, so it is held rather than used.
    assert!(gate.accept(&values, Some(&prev)).is_err());
    // Second look, identical: the gate is satisfied and would hand this to the impedance law.
    assert!(gate.accept(&values, Some(&prev)).is_ok());

    // The envelope refuses it both times, having nothing to be talked out of.
    assert!(first_outside_travel(&values, &measured_envelopes(), &ALL_CORRECTED).is_some());
}

/// A joint with no calibrated travel is not checked. Inventing a limit for it would be worse than
/// the gap: it reads like protection while resting on a number nobody measured.
#[test]
fn a_joint_without_travel_is_not_checked() {
    let envelopes = [None, None, None, None, None, None];
    let values = [0, 4095, 0, 4095, 0, 4095];
    assert_eq!(
        first_outside_travel(&values, &envelopes, &ALL_CORRECTED),
        None
    );
}

/// A joint whose frame is unknown is not checked against its travel either, for the same reason.
///
/// `Operating_Mode` 1 and 3 exist and neither has been measured here, and an unreadable mode is
/// the same kind of ignorance. Guessing wrong does not cost a missed rejection, it rejects every
/// read the joint makes -- which is a torque drop on a loaded arm.
#[test]
fn a_joint_whose_frame_is_unknown_is_not_checked() {
    let frames = [None, None, None, None, None, None];
    let values = [3814, 3814, 325, 517, 3976, 4068];
    assert_eq!(
        first_outside_travel(&values, &measured_envelopes(), &frames),
        None
    );
    // ...but the encoder's own range still applies, since it needs neither frame nor calibration.
    let values = [9999, 3814, 325, 517, 3976, 4068];
    assert_eq!(
        first_outside_travel(&values, &measured_envelopes(), &frames)
            .map(|r| r.verdict),
        Some(TravelVerdict::OffEncoder)
    );
}

/// Every one of this arm's six envelopes crosses the 4095/0 seam once folded into the raw frame,
/// which is structural rather than bad luck: a homing offset is chosen to keep the seam out of the
/// travel in the corrected frame, and that is exactly what puts it inside in the raw one. A
/// non-wrapping comparison would refuse most of every joint's travel.
#[test]
fn every_raw_envelope_on_this_arm_crosses_the_seam() {
    for (motor, envelope) in measured_envelopes().into_iter().enumerate() {
        let Some(envelope) = envelope else { continue };
        let (low, high) = envelope.band(PositionFrame::Raw);
        assert!(low > high, "motor {motor}: raw band {low}-{high} does not wrap");
    }
}
