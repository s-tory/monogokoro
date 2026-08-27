//! PREEMPT_RT setup: CPU pinning and `SCHED_FIFO` priority. Both are best-effort -- failure (no
//! `CAP_SYS_NICE`, not running under an RT-patched kernel, invalid core id) is logged as a
//! warning, not a panic, so this binary stays runnable for local/non-RT development. Everything
//! else needed for real determinism (isolcpus=, nohz_full=, a PREEMPT_RT kernel) is an
//! environment prerequisite documented in README.md, not something this code can provide.

use nix::sched::{sched_setaffinity, CpuSet};
use nix::unistd::Pid;

/// Pins the calling thread to `core_id` and requests `SCHED_FIFO` at `priority` (1-99, higher =
/// more urgent). Both settings are per-*thread*, so each thread that wants them calls this itself.
///
/// `who` names the caller in the log lines. The daemon now has two threads that can ask for this
/// -- the control loop and the cerebellum -- and two identical "pinned to CPU core N" lines with
/// no way to tell which is which is exactly the kind of ambiguity that makes an RT misconfiguration
/// hard to spot.
///
/// A `priority` of 0 means "leave scheduling alone": pin, but do not ask for `SCHED_FIFO`. That is
/// the sensible setting for a thread that wants isolation from the control loop without having any
/// deadline of its own.
pub fn apply_rt_settings(who: &str, core_id: usize, priority: i32) {
    match pin_to_core(core_id) {
        Ok(()) => log::info!("{who}: pinned to CPU core {core_id}"),
        Err(e) => {
            log::warn!(
                "{who}: failed to pin to CPU core {core_id}: {e} (continuing without pinning)"
            )
        }
    }

    if priority == 0 {
        log::info!("{who}: left at default scheduling policy (priority 0)");
        return;
    }

    match set_sched_fifo(priority) {
        Ok(()) => log::info!("{who}: acquired SCHED_FIFO priority {priority}"),
        Err(e) => log::warn!(
            "{who}: failed to set SCHED_FIFO priority {priority}: {e} (continuing at default \
             scheduling policy -- run as root or `setcap cap_sys_nice+ep` on this binary; see \
             README.md)"
        ),
    }
}

fn pin_to_core(core_id: usize) -> nix::Result<()> {
    let mut cpu_set = CpuSet::new();
    cpu_set.set(core_id)?;
    sched_setaffinity(Pid::from_raw(0), &cpu_set)
}

fn set_sched_fifo(priority: i32) -> std::io::Result<()> {
    unsafe {
        let param = libc::sched_param {
            sched_priority: priority,
        };
        let ret = libc::pthread_setschedparam(
            libc::pthread_self(),
            libc::SCHED_FIFO,
            &param as *const libc::sched_param,
        );
        if ret != 0 {
            Err(std::io::Error::from_raw_os_error(ret))
        } else {
            Ok(())
        }
    }
}
