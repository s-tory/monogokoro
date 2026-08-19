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

## 5. Out-of-tree kernel modules

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
