# Running on an Intel GPU (XPU)

LeRobot's own install guide assumes `uv` and CUDA. On an Intel GPU exactly one step differs —
**torch comes from a different index** — and everything else is just translation overhead for
the reader. This page is that one step, plus the traps around it.

## The machine and the day this was taken

|            |                                                                 |
| ---------- | --------------------------------------------------------------- |
| Date       | 2026-09-11                                                      |
| GPU        | Intel Arc 140V (Lunar Lake iGPU, PCI device `0x64A0`, 64 EUs)   |
| OS         | Ubuntu 26.04.1 LTS                                              |
| Kernel     | 7.0.0-31-realtime                                               |
| DRM driver | `xe` (not `i915`)                                               |
| Runtime    | oneAPI Unified Runtime over Level-Zero V2, driver 1.17.39395+13 |
| Python     | 3.12.13 (miniforge / conda-forge)                               |

**Take these again on a different machine.** Every number here is a snapshot of one rig on one
day, and both the driver and the wheels move on a weekly cadence. Do not trust the versions;
trust whether the check in step 6 passes.

## 0. First confirm the kernel sees the GPU

Nothing below matters if this fails.

```bash
lsmod | grep -E '^(i915|xe) ' ; ls /dev/dri/
```

You want a `renderD128` (the number varies). Lunar Lake, Battlemage and later are driven by
`xe`; anything older is `i915`.

## 1. Clone

```bash
git clone https://github.com/s-tory/monogokoro.git && cd monogokoro
```

## 2. Install miniforge

A conda that only ever looks at conda-forge. Run `Miniforge3-Linux-x86_64.sh` from
<https://github.com/conda-forge/miniforge>.

## 3. Create the environment

```bash
conda create -y -n monogokoro python=3.12
conda install -y -n monogokoro ffmpeg
```

`ffmpeg` is needed for video decoding. The conda-forge build also pulls in **`level-zero`,
`intel-gmmlib` and `intel-media-driver`** (1.29.0 / 22.10.0 / 26.1.6 on this rig), so the
minimum Intel runtime is in place without apt.

## 4. Install lerobot — note that this brings the CUDA torch

```bash
conda run -n monogokoro pip install -e .
```

This resolves every dependency. But **the `torch` on PyPI is the CUDA build**, so `torch.xpu`
does not work yet. Step 5 replaces it.

## 5. Replace torch with the XPU build — this is the step that differs

```bash
conda run -n monogokoro pip install --index-url https://download.pytorch.org/whl/xpu \
  --force-reinstall torch torchvision
```

This rig landed on `torch 2.11.0+xpu` / `torchvision 0.26.0+xpu` (`torch.version.xpu` reports
`20250302`). The wheels carry the whole Intel runtime with them, so **oneAPI does not have to be
installed system-wide**:

```
dpcpp-cpp-rt  intel-sycl-rt  intel-cmplr-lib-rt  intel-cmplr-lib-ur  intel-opencl-rt
intel-openmp  intel-pti  mkl  onemkl-sycl-{blas,dft,lapack,rng,sparse}
tcmlib  umf  impi-rt  triton-xpu
```

(2025.3.x for the Intel packages, 2025.3.1 for oneMKL, on this rig.)

**The swap leaves CUDA debris behind.** Only torch itself is replaced, so the `nvidia-*`
packages (15 of them here) plus `cuda-toolkit`, `cuda-bindings` and `cuda-pathfinder` are
orphaned — a closed island that depends only on itself and is never loaded again. **Deleting it
on this rig on 2026-08-31 took the environment from 13G to 11G** (2.7GB). A bf16 matmul on
`torch.xpu`, `triton-xpu`, and `import lerobot` were all verified afterwards.

## 6. Verify

```bash
conda run -n monogokoro python -c "
import torch
print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())
print(torch.xpu.get_device_name(0))
"
```

What this rig prints:

```
2.11.0+xpu True 1
Intel(R) Arc(TM) Graphics
```

Pass `--policy.device=xpu` to training and evaluation. It is usually detected automatically and
can be omitted.

---

## Traps

### Do not `pip uninstall triton`

This environment holds `triton 3.6.0` (the CUDA build) and `triton-xpu 3.7.0` **sharing one
`triton/` directory**, with 436 files overlapping. The active one is 3.7.0. Uninstalling
`triton` takes the shared files with it, **breaking triton-xpu and killing `torch.compile`**.
All that actually remains of 3.6.0 is its dist-info, so the only real symptom is that `pip`
misreports what is installed. Leaving it alone is correct.

### The CUDA and XPU builds of torch cannot share an environment

Both are packaged as `torch`, and whichever is installed last wins. `torch 2.11.0+xpu` ships no
`libtorch_cuda.so` at all (only `libtorch_cpu.so` and `libtorch_xpu.so`), and
`torch.version.cuda` is `None`. Installing the NVIDIA driver does not change this —
`torch.cuda.is_available()` stays False. **If you need both, build two environments.**

### conda-forge's `level-zero` shadows the system loader

Measured on this rig on 2026-09-11, by running one XPU matmul and reading `/proc/self/maps`:

```
env      lib/libze_loader.so.1.29.0                      <- the Level Zero loader
env      lib/libsycl.so.8, lib/libOpenCL.so.1
system   /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395   <- the actual GPU driver
system   /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
```

The loader comes from the environment and the driver from the system. That loader is
conda-forge's `level-zero 1.29.0`, pulled in by the `ffmpeg` in step 3 — pip ships none. But apt
carries one too (`libze1 1.32.0` here), and the environment's `lib/` comes first on the search
path, so **the older conda copy is the one that gets loaded**.

**Whether this costs anything is unmeasured.** It works. A Level Zero loader is meant to drive
other generations of driver. But it is a version pin nobody asked for, and worth knowing about
before blaming the driver for something.

### An OOM on an integrated GPU takes the desktop with it

On an iGPU, GPU memory and system memory are the same pool, so a training OOM sends the OOM
killer after the compositor and `dbus-daemon`. **On 2026-08-30 this rig lost its whole GNOME
session that way** — not a machine reboot, but everything on screen went. A discrete card has
isolated VRAM and would have raised `torch.OutOfMemoryError` instead.

Wrap long runs in a cgroup:

```bash
systemd-run --user --scope -p MemoryMax=20G lerobot-train ...
```

---

## What cannot be written yet

**A two-GPU setup is unverified.** This rig has one iGPU and nothing else. The following waits
on the eGPU (an Intel Arc B580) arriving, and will be measured before it is written:

- How `xpu:0` and `xpu:1` are assigned. **With two devices, `xpu:0` is not necessarily the
  iGPU.** Enumeration order follows the PCI ordering and the driver's initialization order, and
  can swap across a reboot or a dock being unplugged. **Checking whether the devices can be told
  apart by number at all is the first thing to do once the second one is in.**
- Whether a device can be pinned by PCI address — the equivalent of the `by-path` escape used
  for the cameras. Not investigated.
- Using `ZE_AFFINITY_MASK` or similar to bind a layer to a specific device.

The B580 is an Intel card, so it joins **this XPU environment as a second device** rather than
going in a CUDA environment.
