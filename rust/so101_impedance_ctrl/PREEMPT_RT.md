# PREEMPT_RT host setup

Everything here is **environment setup on the machine wired to the arm**, not something this crate
installs or validates. `so101_impedance_ctrl` runs without any of it (CPU pinning and `SCHED_FIFO`
degrade to a logged warning), you just don't get deterministic timing.

Findings below marked "reference machine" were measured on a ThinkPad X1 Carbon Gen 13 (Core Ultra
7 258V, 4 P-cores + 4 LP E-cores) running `7.0.0-28-realtime`.

## 1. Kernel and boot parameters

You need a PREEMPT_RT-patched (or at least low-latency) kernel, plus boot parameters that isolate
the target core from the scheduler, timer tick, RCU callbacks, **and device interrupts**. For core
3, add to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

```
isolcpus=managed_irq,domain,3 nohz_full=3 rcu_nocbs=3 irqaffinity=0-2,4-7
```

then `sudo update-grub && sudo reboot`. To isolate a different core, change the index in all three
of `isolcpus`/`nohz_full`/`rcu_nocbs`, list every _other_ core in `irqaffinity`, and pass the same
index as `--cpu-core`.

- `domain` preserves the scheduler isolation that plain `isolcpus=N` gives.
- `managed_irq` steers driver-managed per-CPU IRQs away from the isolated core.
- `irqaffinity=` keeps all remaining (unmanaged) IRQs off it.

### Do not use a bare `isolcpus=N nohz_full=N rcu_nocbs=N`

Those three isolate the scheduler, timer tick and RCU callbacks only -- they do **not** move device
interrupts. Drivers pin per-CPU MSI-X queues (NVMe completion queues, `iwlwifi` RX queues, ...) at
probe time, so an interrupt-heavy queue keeps firing on the "isolated" core while nothing is
scheduled there to service it promptly. Under PREEMPT_RT that can stall storage I/O or trip a
device firmware watchdog and **hard-freeze the machine seconds into boot**, long before the display
manager starts -- the journal just ends mid-boot with no panic message.

Reference machine: a bare `isolcpus=2 nohz_full=2 rcu_nocbs=2` froze the box ~6 s into every boot.
Core 2 owned `nvme0q4` (a hot NVMe completion queue) plus an `iwlwifi` RX queue, and carried ~6x
core 3's interrupt count; an `iwlwifi ... NMI_INTERRUPT_UMAC_FATAL` fired ~2 s before each hang and
appeared in _no_ other boot. Core 3 survived only because it happened to hold the near-idle NVMe
_admin_ queue -- luck, not safety. Add `managed_irq` + `irqaffinity` whichever core you pick.

Picking an E-core instead does not help: on the reference machine every CPU 0-7 had exactly one
`nvme` queue and one `iwlwifi` queue pinned to it. P-cores are marginally better for isolation
anyway (private L2; the LP E-cores share one L2 across all four and have no L3).

## 2. Verifying the isolation actually took

What matters is not which IRQs are _bound_ to the isolated core, but which ones actually _fire_
there. `/proc/interrupts` counts are **cumulative since boot**, so a nonzero total may just be
history from before a fix was applied -- sample twice and look at the delta:

```bash
# Report only device IRQs whose CPU3 count is still GROWING. Empty output == clean.
# Column 5 is CPU3; the `$1 ~ /^[0-9]+:$/` filter skips the architecture-specific summary rows
# at the bottom of the file (LOC/CAL/TRM/...), which are inherently per-CPU and unmovable.
snap() { awk '$1 ~ /^[0-9]+:$/ {gsub(":","",$1); print $1, $5, $NF}' /proc/interrupts; }
snap > /tmp/irq0; sleep 5; snap > /tmp/irq1
join /tmp/irq0 /tmp/irq1 | awk '$4 > $2 {print $1, $3, "+" ($4-$2)}'
```

Exercise the suspect device while sampling (e.g. `ping` the gateway for a Wi-Fi queue), or the
delta is zero simply because the device was idle.

An IRQ _bound_ to the isolated core with a **zero** delta is fine -- that is `managed_irq` working.
NVMe allocates one completion queue per CPU, so `nvme0qN` for the isolated core keeps a
core-N-only mask with nothing to fall back to, but nothing is scheduled there so it never fires.

## 3. Driver-pinned IRQs that survive `irqaffinity=`

`irqaffinity=` only seeds the _default_ mask. Drivers that call `irq_set_affinity_hint()` per queue
-- `iwlwifi` pins one RX queue per CPU -- still land on the isolated core. On the reference machine
`iwlwifi:queue_4` kept firing on core 3 with the full parameter set above. Move it at runtime:

```ini
# /etc/systemd/system/rt-irq-affinity.service
[Unit]
Description=Move device IRQs off the isolated RT core
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'for f in /proc/irq/*/smp_affinity_list; do echo 0-2,4-7 > "$f" 2>/dev/null || true; done'

[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now rt-irq-affinity.service`, then re-run the delta check. Verified
working on the reference machine: `iwlwifi:queue_4` moved from core 3 to core 0 and the core-3
delta went to zero under `ping` load.

This works only for IRQs whose affinity the driver set via `irq_set_affinity_hint()`. Check
`/proc/irq/N/affinity_hint`: nonzero means userspace can override it; zero means kernel-managed and
the write is rejected with `EIO` (harmless, silenced above -- those are what `managed_irq` handles).

Two ways it silently stops working:

- **`irqbalance`**, if installed, periodically redistributes IRQs and undoes the unit. Don't run it
  on the robot host, or ban the isolated core (`IRQBALANCE_BANNED_CPULIST=3` in
  `/etc/default/irqbalance`).
- The unit is `oneshot`, so a driver that **re-pins later** -- Wi-Fi suspend/resume, interface
  down/up, firmware restart, module reload -- gets its affinity back. Re-run the delta check after
  any such event and `systemctl restart rt-irq-affinity.service` if needed.

Note this _redirects_ interrupts rather than eliminating them: the device generates the same load,
just serviced on housekeeping cores. Disabling Wi-Fi and using Ethernet removes the source at the
root and needs no upkeep.

## 4. Scheduling privileges

Either run the daemon as root, or grant it `CAP_SYS_NICE`:

```bash
sudo setcap cap_sys_nice+ep ./target/release/so101_impedance_ctrl
```

plus a realtime `ulimit -r` / `/etc/security/limits.d/*.conf rtprio` entry for the invoking user.
Also disable CPU frequency scaling / turbo boost on the isolated core for consistent loop timing.

## 5. The cerebellum's GPU thread: do not give it a second isolated core

The obvious next move once one core is isolated is to isolate a second for the cerebellum's Vulkan
thread. Don't. It buys nothing here, and it costs a P-core.

- **There is no compute queue to escape to.** `vulkaninfo` on the reference machine reports the
  iGPU with **one queue family, `queueCount = 1`**, carrying `GRAPHICS | COMPUTE | TRANSFER`.
  Compute submissions share a queue with the desktop compositor. No CPU-side scheduling decision
  changes what that queue is doing.
- **The GPU's service path is on housekeeping cores by construction.** Driver workqueues, the DRM
  scheduler and GPU completion interrupts all run wherever the kernel puts them -- and §1's whole
  purpose is to steer interrupts _away_ from the isolated core. So a `SCHED_FIFO` thread pinned to
  an isolated core would submit deterministically and then wait on a fence completed by threads
  that are not isolated at all.
- **The thread has no deadline.** The load it predicts is quasi-static, so a feedforward a few
  milliseconds late is a feedforward that is still correct.

What actually has to be true is that the cerebellum can never preempt or block the control loop,
and `isolcpus` on the control loop's core already guarantees that. So:

```bash
--cerebellum-cpu-core 1     # a housekeeping P-core; the daemon refuses --cpu-core's core
--cerebellum-priority 0     # normal scheduling (the default)
```

If you do raise `--cerebellum-priority`, it must stay below `--priority`; the daemon refuses
otherwise, so that a mistake in core assignment cannot let the cerebellum outrank the reflex.

Verify the split is real the same way §2 verifies IRQ isolation -- by looking at the loop timing
summary before and after enabling the cerebellum. If `--cerebellum-backend gpu` changes the control
loop's `max` or its overrun count at all, the two are not actually decoupled and the core
assignment is wrong.

## 6. Out-of-tree kernel modules

On PREEMPT_RT most spinlocks become sleeping locks, so a module written for a non-RT kernel can
hold an atomic context across one. It then floods the log with `BUG: scheduling while atomic` and
permanently taints the kernel -- and the latency guarantees the RT kernel exists to provide no
longer hold. (Observed here with VirtualBox's `vboxdrv`; unrelated to the isolation parameters, it
reproduces with no `isolcpus=` at all.)

```bash
journalctl -b -k | grep -c "scheduling while atomic"   # want 0
cat /proc/sys/kernel/tainted                            # bits 12 (O) and 13 (E) are the ones here
```

Taint bit 12 (`4096`) is `O` = out-of-tree, bit 13 (`8192`) is `E` = unsigned; bit 9 (`512`, `W`)
just records that a `BUG`/`WARN` fired. Find the culprit in the `Modules linked in:` line of the
trace -- the loading module is tagged `(OE+)`. Taint flags are sticky: unloading does not clear
them, only a reboot does.

Whether it actually costs you latency is measurable, so measure before ripping anything out:

```bash
sudo apt install rt-tests
sudo cyclictest --smp -p99 -a3 -t1 -m -D 60   # -a3 = pin to the isolated core
```

Watch the `Max` column: for a 1 kHz loop you want worst case comfortably under ~100 us -- tens of
microseconds is healthy, millisecond spikes are not. If the numbers are fine, leaving the module
loaded is a reasonable call; if not, stop it autoloading or uninstall it on the robot host.

### Choosing hardware whose driver is in-tree

The axis that matters is not open source, it is merged. A module that lives in the mainline tree
is built together with the RT kernel, signed with its key, and carried by every kernel update. A
module built by DKMS is none of those things, and has to be rebuilt against each new kernel that
appears.

What that looks like on this machine, on 2026-09-01, trying to put an NVIDIA card next to the
realtime kernel -- read out of `/var/log/apt/history.log`:

```
11:53  install  linux-headers-7.0.0-30-realtime
12:06  install  nvidia-driver-595-open
12:18  purge    nvidia (all of it)
12:18  purge    linux-headers-7.0.0-28{,-generic,-realtime}
12:30  purge    linux-headers-7.0.0-30-realtime + linux-realtime-headers-7.0.0-30
12:30  install  nvidia-driver-595-open        <- five seconds after the RT headers went away
13:22  install  nvidia-driver-595             <- the proprietary build, after -open
13:31  purge    nvidia (all of it)
13:31  install  linux-headers-7.0.0-30-realtime
```

The RT headers and the NVIDIA packages are alternating. Removing the realtime headers five
seconds before reinstalling the driver is DKMS being steered away from the RT kernel so it would
build for the generic one instead. Both the `-open` and the proprietary build were tried. The
last two lines are how it ended: **the driver was dropped and the realtime kernel was kept.** The
`rc nvidia-dkms-595-open` still in `dpkg -l` is what is left of that afternoon.

**This is one card on one machine on one day, not a claim about NVIDIA in general.** What it
cost is recorded because the cost is the point: an afternoon, and a GPU that could not be used on
the robot host.

The contrast is worth having in the same place. Virtualization on this host is qemu/KVM, and:

```bash
modinfo kvm_intel | grep -E '^(filename|intree|signer)'
# filename: /lib/modules/7.0.0-31-realtime/kernel/arch/x86/kvm/kvm-intel.ko.zst
# intree:   Y
# signer:   Build time autogenerated kernel key
cat /proc/sys/kernel/tainted   # 0
```

Inside the RT kernel's own module tree, built with it, signed by it. Nothing to rebuild and
nothing to taint. Moving off VirtualBox and onto qemu/KVM here was not a preference between
virtualization stacks; it was moving from out-of-tree to in-tree, and the `0` above is the
result.

**Check this before buying, not after.** `modinfo <module> | grep intree` on a machine that
already has the hardware, or whether the driver appears under `drivers/` in the mainline tree at
all, answers it in one line.

## 7. Knobs that do nothing on this link

Once the isolation above is in place the control loop spends nearly every tick blocked on the
servo link, so the next instinct is to go tune the serial port. There is nothing there.

**`setserial /dev/ttyACM0 low_latency` does not change the round trip.** The flag is a
`serial_core` / `usbserial` feature; the SO-101's bridge is a CH343 in CDC mode, driven by
`cdc_acm`, which has no such flag to set.

What makes it worth a section is _how_ it fails: **silently, and it cannot be caught by
inspection.** `cdc_acm` accepts the `TIOCSSERIAL` and returns success without storing anything --
even unprivileged -- so the command prints no error. And its `TIOCGSERIAL` zeroes `flags`
unconditionally, so reading the port back afterwards shows the flag clear whether or not anything
was stored. Neither the command's exit status nor the read-back carries information. Only the
round trip does:

|                   | mean     | p50   | p90   | p99   |
| ----------------- | -------- | ----- | ----- | ----- |
| `low_latency` off | 334.9 us | 329.6 | 370.9 | 475.8 |
| `low_latency` on  | 334.0 us | 330.0 | 368.5 | 466.5 |

**0.9 us apart at 1.0 sigma**, against 15 us of scatter between blocks -- no effect. Measured
2026-09-12, follower arm on `/dev/ttyACM0` (CH343 `1a86:55d3`, `cdc_acm`, 1 Mbaud), 4800
`Present_Position` reads per condition, round-robin over the six motors. Both sit above the ~256 us
the daemon itself measures because this probe is Python at normal priority and flushes the input
buffer before every read -- an overhead both conditions carry equally, which is all a comparison
needs.

The order had to be alternated to get that number. Running the flag-off block first every time gave
`-3.6 us` in the flag's favour, which is 3.5 sigma and looks like a result; it was warm-up drift
over the run, and it disappeared when half the blocks ran flag-on first. A knob that does nothing
will still look like it does something if it is always measured second.

The FTDI `latency_timer` (16 ms by default, and a real win when it applies) is the knob people
remember. It is an `ftdi_sio` feature and these bridges do not have it. The ~256 us per transaction
the daemon sees against ~160 us of wire time at 1 Mbaud is the USB round trip itself -- host
scheduling, not tty buffering -- and no userspace flag reaches it.
