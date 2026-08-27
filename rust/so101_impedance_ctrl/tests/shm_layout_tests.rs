//! Shared-memory struct-layout sanity checks and a concurrent seqlock stress test -- no real
//! shared memory segment needed, this exercises the same types in-process.

use std::cell::UnsafeCell;
use std::mem::size_of;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::thread;

use so101_impedance_ctrl::shm::{seqlock_read, seqlock_write, InputData, ShmLayout, NUM_MOTORS};

#[test]
fn shm_layout_is_nonzero_sized_plain_old_data() {
    // A coarse sanity check: if this ever becomes 0 (e.g. an accidental empty struct after a
    // refactor) something has gone badly wrong with the cross-language layout.
    assert!(size_of::<ShmLayout>() > 0);
}

/// Wraps `InputData` behind an `UnsafeCell` to emulate the shared-memory access pattern (a
/// `&AtomicU32` seq field living alongside a raw, non-atomic data region that multiple threads
/// read/write through raw pointers) without needing an actual OS shared-memory segment.
struct SharedInput {
    seq: AtomicU32,
    data: UnsafeCell<InputData>,
}
// SAFETY: all access to `data` in this test goes through `seqlock_read`/`seqlock_write`, which
// synchronize via `seq` -- the same contract the real cross-process shared-memory segment relies
// on.
unsafe impl Sync for SharedInput {}

#[test]
fn seqlock_reader_never_observes_a_torn_write() {
    let shared = Arc::new(SharedInput {
        seq: AtomicU32::new(0),
        data: UnsafeCell::new(InputData::default()),
    });

    let writer_shared = Arc::clone(&shared);
    let writer = thread::spawn(move || {
        for i in 0..20_000u64 {
            let value = i as f32;
            // SAFETY: seqlock_write's fences make this exclusive-for-the-duration-of-the-closure
            // with respect to any seqlock_read observing an even sequence number.
            unsafe {
                seqlock_write(&writer_shared.seq, &mut *writer_shared.data.get(), |d| {
                    // Every field gets a value derived from the same counter -- a torn read
                    // would show mismatched fields between them.
                    d.timestamp_mono_ns = i;
                    d.target_pos = [value; NUM_MOTORS];
                    d.k_gain = [-value; NUM_MOTORS];
                });
            }
        }
    });

    let reader_shared = Arc::clone(&shared);
    let reader = thread::spawn(move || {
        let mut stable_reads = 0;
        for _ in 0..20_000 {
            // `read_fn` only extracts data -- it must NOT assert here, since it can run during a
            // still-in-flight write, before `seqlock_read` has confirmed stability. Validation
            // happens below, only once `seqlock_read` returns `Some` (a confirmed-stable read);
            // `None` (retries exhausted under contention) is a legitimate outcome the caller
            // must handle by falling back to its last known-good value, not by trusting an
            // unstable read -- see the doc comment on `seqlock_read` and `main.rs`'s
            // `last_good_input`.
            // SAFETY: reads are gated by the same seqlock the writer uses.
            let snapshot = unsafe {
                seqlock_read(
                    &reader_shared.seq,
                    &*reader_shared.data.get(),
                    |d| (d.timestamp_mono_ns, d.target_pos, d.k_gain),
                    64,
                )
            };
            if let Some((timestamp, target_pos, k_gain)) = snapshot {
                let expected = timestamp as f32;
                assert_eq!(
                    target_pos, [expected; NUM_MOTORS],
                    "torn read on target_pos"
                );
                assert_eq!(k_gain, [-expected; NUM_MOTORS], "torn read on k_gain");
                stable_reads += 1;
            }
        }
        // Under realistic contention the seqlock should succeed the great majority of the time;
        // if it never once returns a stable snapshot, the retry budget or seqlock itself is
        // broken, not just "occasionally contended".
        assert!(
            stable_reads > 0,
            "seqlock_read never once returned a stable snapshot"
        );
    });

    writer.join().unwrap();
    reader.join().unwrap();
}

#[test]
fn shm_layout_size_matches_the_python_mirror() {
    // `shm_client.py`'s `ctypes.Structure` mirror is maintained by hand, and `LAYOUT_VERSION` only
    // catches the case where someone remembers to bump it. This catches the case where they do not
    // -- or where the two sides gain the same fields in a different order, which produces matching
    // versions and mismatched bytes.
    //
    // Recompute after any layout change with:
    //   python -c "import ctypes; from lerobot.robots.so101_impedance_follower.shm_client import \
    //              ShmLayout; print(ctypes.sizeof(ShmLayout))"
    assert_eq!(
        std::mem::size_of::<ShmLayout>(),
        320,
        "ShmLayout changed size -- update shm_client.py's mirror and this number together, and \
         bump LAYOUT_VERSION"
    );
}
