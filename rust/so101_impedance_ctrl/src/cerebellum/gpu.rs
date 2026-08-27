//! Vulkan compute backend for the cerebellum, on the integrated GPU.
//!
//! Everything here is one thread's private property. The cerebellum thread creates the device,
//! records one command buffer, and then per step does nothing but write a small host-visible
//! buffer, submit, and wait. No descriptor updates, no allocation, no command re-recording on the
//! steady path.
//!
//! # Why an iGPU makes this simple
//!
//! A discrete card would need staging buffers and an explicit transfer queue for every upload and
//! readback. An integrated GPU shares the CPU's memory controller, so Vulkan reports memory types
//! that are `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` at once: every buffer is mapped once at
//! init and stays mapped, the host writes mossy fibres straight into the memory the shader reads,
//! and there is no copy anywhere in the per-step path. The unified memory that makes an iGPU
//! unimpressive at graphics is exactly what makes it good at a small network stepped at kilohertz.
//!
//! # What this does not do
//!
//! It does not try to be real-time. On this hardware Vulkan reports **one queue family with one
//! queue**, shared with graphics -- there is no separate compute queue to submit into, so work
//! here queues behind whatever the desktop compositor is doing, and no amount of CPU isolation
//! changes that. The GPU's own service path (driver workqueues, the DRM scheduler, completion
//! interrupts) also runs on housekeeping cores by construction, since the RT core's whole point is
//! that interrupts are steered away from it.
//!
//! That is why the caller runs this on its own thread and the control loop never waits on it:
//! see `mod.rs`.

use std::ffi::CStr;
use std::io::Cursor;

use ash::{vk, Device, Entry, Instance};

use super::net::{GranuleParams, GC_FAN_IN, MF_DIM, NUM_OUTPUTS};

/// Local workgroup size, matching `layout(local_size_x = 256)` in all three shaders.
const WORKGROUP_SIZE: u32 = 256;

/// How long to wait for a submission before declaring the device unusable.
///
/// A step is microseconds of actual work; anything approaching this bound means the driver or the
/// device is wedged, not that the network is slow. Bounded rather than infinite because the
/// alternative is a thread parked forever inside `vkWaitForFences` holding a mapped device, which
/// is indistinguishable from a hang and cannot be shut down cleanly.
const FENCE_TIMEOUT_NS: u64 = 500_000_000;

/// Mirrors the `Params` block in `shaders/common.glsl`. `#[repr(C)]` against std430, which for a
/// block of scalars and a float array is the same layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuParams {
    gc_dim: u32,
    learn: u32,
    theta: f32,
    trace_decay: f32,
    rate: f32,
    leak: f32,
    cf: [f32; NUM_OUTPUTS],
}

/// Mirrors the `Readout` block in `shaders/common.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuReadout {
    ff: [f32; NUM_OUTPUTS],
    active_count: u32,
}

/// Binding indices, in the order `common.glsl` declares them.
mod binding {
    pub const PARAMS: u32 = 0;
    pub const MOSSY: u32 = 1;
    pub const GC_IDX: u32 = 2;
    pub const GC_WEIGHT: u32 = 3;
    pub const GC_BIAS: u32 = 4;
    pub const GC_ACT: u32 = 5;
    pub const TRACE: u32 = 6;
    pub const WEIGHTS: u32 = 7;
    pub const READOUT: u32 = 8;
    pub const COUNT: usize = 9;
}

struct MappedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

impl MappedBuffer {
    /// Copies `data` into the mapping.
    ///
    /// Copy in and copy out, rather than handing back a `&mut [T]` over the mapping: a reference
    /// into memory the GPU also writes is only sound while no submission is in flight, and that is
    /// an invariant no signature can express. Every transfer here is at most a few hundred bytes
    /// on the hot path (the whole point of the design is that the *network* stays on the device),
    /// so the copies cost nothing that is worth the aliasing footgun.
    ///
    /// # Safety
    /// `T` must match the shader-side layout of this binding, and no submission touching this
    /// buffer may be in flight.
    unsafe fn write_slice<T: Copy>(&self, data: &[T]) {
        debug_assert!(std::mem::size_of_val(data) <= self.size as usize);
        std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr as *mut T, data.len());
    }

    /// # Safety
    /// As [`Self::write_slice`].
    unsafe fn write_one<T: Copy>(&self, value: T) {
        debug_assert!(std::mem::size_of::<T>() <= self.size as usize);
        std::ptr::write_unaligned(self.ptr as *mut T, value);
    }

    /// # Safety
    /// As [`Self::write_slice`].
    unsafe fn read_one<T: Copy>(&self) -> T {
        debug_assert!(std::mem::size_of::<T>() <= self.size as usize);
        std::ptr::read_unaligned(self.ptr as *const T)
    }

    /// # Safety
    /// As [`Self::write_slice`].
    unsafe fn read_vec<T: Copy>(&self, len: usize) -> Vec<T> {
        debug_assert!(std::mem::size_of::<T>() * len <= self.size as usize);
        let mut out = Vec::with_capacity(len);
        std::ptr::copy_nonoverlapping(self.ptr as *const T, out.as_mut_ptr(), len);
        out.set_len(len);
        out
    }
}

pub struct GpuNet {
    // Declaration order is destruction order for the plain fields, but Vulkan handles are freed
    // explicitly in `Drop` (below) in dependency order, so this only needs to keep the loader and
    // instance alive.
    _entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,

    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    modules: Vec<vk::ShaderModule>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    buffers: Vec<MappedBuffer>,

    gc_dim: usize,
    pub device_name: String,
}

// SAFETY: every handle and mapped pointer in here is used from exactly one thread -- the
// cerebellum thread, which takes ownership at construction and never shares it. `GpuNet` is
// `!Sync` by omission (no `Sync` impl), so it can be moved to that thread but never referenced
// from two.
unsafe impl Send for GpuNet {}

impl GpuNet {
    /// Brings up Vulkan, uploads the fixed granule connectivity, and records the command buffer.
    ///
    /// Every failure path here returns `Err` rather than panicking: a machine with no Vulkan
    /// driver, a headless CI runner, or a GPU in a bad state must all leave the daemon running the
    /// arm. The caller degrades to no feedforward and says so loudly.
    pub fn new(params: &GranuleParams, prefer_integrated: bool) -> Result<Self, String> {
        assert_eq!(
            GC_FAN_IN, 4,
            "shaders/common.glsl hardcodes GC_FAN_IN = 4; change both together"
        );

        // SAFETY: the whole Vulkan API is unsafe by construction. Each call below is guarded by
        // the usual rules -- handles are used only while alive, slices outlive the calls that
        // borrow them -- and any failure is turned into an `Err` before it can be acted on.
        unsafe {
            let entry = Entry::load().map_err(|e| {
                format!("no Vulkan loader ({e}) -- install a Vulkan ICD (mesa-vulkan-drivers)")
            })?;

            let app_name = c"so101_impedance_ctrl";
            let app_info = vk::ApplicationInfo::default()
                .application_name(app_name)
                .api_version(vk::make_api_version(0, 1, 1, 0));
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app_info),
                    None,
                )
                .map_err(|e| format!("vkCreateInstance failed: {e}"))?;

            let (physical, queue_family, device_name) =
                match select_device(&instance, prefer_integrated) {
                    Ok(v) => v,
                    Err(e) => {
                        instance.destroy_instance(None);
                        return Err(e);
                    }
                };
            log::info!(
                "cerebellum: using Vulkan device '{device_name}' (queue family {queue_family})"
            );

            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                    None,
                )
                .map_err(|e| {
                    instance.destroy_instance(None);
                    format!("vkCreateDevice failed: {e}")
                })?;
            let queue = device.get_device_queue(queue_family, 0);

            // From here on a failure has to tear down what already exists, so the rest is built in
            // a helper and unwound as a unit.
            match Self::build(
                entry,
                instance,
                physical,
                device,
                queue,
                queue_family,
                params,
                device_name,
            ) {
                Ok(net) => Ok(net),
                Err(e) => Err(e),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn build(
        entry: Entry,
        instance: Instance,
        physical: vk::PhysicalDevice,
        device: Device,
        queue: vk::Queue,
        queue_family: u32,
        params: &GranuleParams,
        device_name: String,
    ) -> Result<Self, String> {
        let gc_dim = params.gc_dim;
        let mem_props = instance.get_physical_device_memory_properties(physical);

        let sizes = buffer_sizes(gc_dim);
        let mut buffers = Vec::with_capacity(binding::COUNT);
        for (i, &size) in sizes.iter().enumerate() {
            match create_mapped_buffer(&device, &mem_props, size) {
                Ok(b) => buffers.push(b),
                Err(e) => {
                    for b in &buffers {
                        device.destroy_buffer(b.buffer, None);
                        device.free_memory(b.memory, None);
                    }
                    device.destroy_device(None);
                    instance.destroy_instance(None);
                    return Err(format!("binding {i}: {e}"));
                }
            }
        }

        // Upload the fixed connectivity once. It never changes again -- it is not learned.
        buffers[binding::GC_IDX as usize].write_slice(&params.idx);
        buffers[binding::GC_WEIGHT as usize].write_slice(&params.weight);
        buffers[binding::GC_BIAS as usize].write_slice(&params.bias);
        // Purkinje weights, granule activity and the eligibility trace all start at zero: an
        // untrained cerebellum must contribute exactly nothing, so that turning it on cannot
        // change how the arm behaves until it has learned something.
        for b in [
            binding::WEIGHTS as usize,
            binding::GC_ACT as usize,
            binding::TRACE as usize,
            binding::READOUT as usize,
            binding::PARAMS as usize,
            binding::MOSSY as usize,
        ] {
            std::ptr::write_bytes(buffers[b].ptr, 0, buffers[b].size as usize);
        }

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding::COUNT as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let descriptor_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {e}"))?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(binding::COUNT as u32)];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorPool failed: {e}"))?;

        let set_layouts = [descriptor_layout];
        let descriptor_set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
            .map_err(|e| format!("vkAllocateDescriptorSets failed: {e}"))?[0];

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = (0..binding::COUNT)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i]))
            })
            .collect();
        device.update_descriptor_sets(&writes, &[]);

        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;

        // Compiled from `shaders/*.comp` by build.rs, so a shader edit cannot ship without its
        // binary being rebuilt.
        let spirv: [&[u8]; 4] = [
            include_bytes!(concat!(env!("OUT_DIR"), "/granule.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/normalise.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/purkinje.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/learn.spv")),
        ];
        let entry_point = c"main";
        let mut modules = Vec::with_capacity(spirv.len());
        for blob in spirv {
            let code = ash::util::read_spv(&mut Cursor::new(blob))
                .map_err(|e| format!("malformed SPIR-V produced by build.rs: {e}"))?;
            modules.push(
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                    .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?,
            );
        }

        let stages: Vec<vk::PipelineShaderStageCreateInfo> = modules
            .iter()
            .map(|&m| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(m)
                    .name(entry_point)
            })
            .collect();
        let pipeline_infos: Vec<vk::ComputePipelineCreateInfo> = stages
            .iter()
            .map(|&stage| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)
            })
            .collect();
        let pipelines = device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
            .map_err(|(_, e)| format!("vkCreateComputePipelines failed: {e}"))?;

        let command_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
                None,
            )
            .map_err(|e| format!("vkCreateCommandPool failed: {e}"))?;
        let command_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("vkAllocateCommandBuffers failed: {e}"))?[0];

        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("vkCreateFence failed: {e}"))?;

        let net = Self {
            _entry: entry,
            instance,
            device,
            queue,
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
            pipeline_layout,
            pipelines,
            modules,
            command_pool,
            command_buffer,
            fence,
            buffers,
            gc_dim,
            device_name,
        };
        net.record_command_buffer()?;
        Ok(net)
    }

    /// Records the four dispatches once. Nothing in the recording depends on per-step data -- that
    /// all lives in the mapped `Params` buffer -- so this is never re-recorded.
    ///
    /// The barriers are load-bearing, every one of them for correctness rather than performance:
    ///
    /// * granule -> normalise, because the reduction has to see the whole field.
    /// * normalise -> Purkinje, because the readout must see the *normalised* code and the trace
    ///   the same step wrote.
    /// * Purkinje -> learn, because the readout must see the weights **before** plasticity updates
    ///   them. Without it the two dispatches race and the feedforward becomes a mixture of old and
    ///   new weights that differs run to run.
    unsafe fn record_command_buffer(&self) -> Result<(), String> {
        let d = &self.device;
        let cmd = self.command_buffer;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
            .map_err(|e| format!("vkBeginCommandBuffer failed: {e}"))?;
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[self.descriptor_set],
            &[],
        );

        let groups = self.gc_dim.div_ceil(WORKGROUP_SIZE as usize) as u32;
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        let compute_barrier = || {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            )
        };

        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[0]);
        d.cmd_dispatch(cmd, groups, 1, 1);
        compute_barrier();

        // Exactly one workgroup: the L2 reduction has to be visible to every element it rescales,
        // and shared memory plus `barrier()` is the only synchronisation available inside a
        // dispatch.
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[1]);
        d.cmd_dispatch(cmd, 1, 1, 1);
        compute_barrier();

        // One workgroup per Purkinje cell; each reduces the whole granule field.
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[2]);
        d.cmd_dispatch(cmd, NUM_OUTPUTS as u32, 1, 1);
        compute_barrier();

        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[3]);
        d.cmd_dispatch(cmd, groups, 1, 1);

        d.end_command_buffer(cmd)
            .map_err(|e| format!("vkEndCommandBuffer failed: {e}"))
    }

    pub fn gc_dim(&self) -> usize {
        self.gc_dim
    }

    /// Runs one full step: inference, then plasticity if `learn` is set.
    ///
    /// Returns the Purkinje output (feedforward duty, unclamped) and the fraction of granule cells
    /// that fired, which the caller's Golgi integrator uses to adjust `theta`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        mf: &[f32; MF_DIM],
        theta: f32,
        trace_decay: f32,
        cf: Option<&[f32; NUM_OUTPUTS]>,
        rate: f32,
        leak: f32,
    ) -> Result<([f32; NUM_OUTPUTS], f32), String> {
        // SAFETY: all buffers are host-coherent and permanently mapped; this thread is the only
        // writer, and the GPU is idle between submissions because the previous step waited on its
        // fence before returning.
        unsafe {
            let params = GpuParams {
                gc_dim: self.gc_dim as u32,
                learn: u32::from(cf.is_some()),
                theta,
                trace_decay,
                rate,
                leak,
                cf: cf.copied().unwrap_or([0.0; NUM_OUTPUTS]),
            };
            self.buffers[binding::PARAMS as usize].write_one(params);
            self.buffers[binding::MOSSY as usize].write_slice(mf);
            // The shader accumulates into this with atomics, so it has to start from zero.
            self.buffers[binding::READOUT as usize].write_one(GpuReadout::default());

            let command_buffers = [self.command_buffer];
            let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|e| format!("vkQueueSubmit failed: {e}"))?;
            let fences = [self.fence];
            self.device
                .wait_for_fences(&fences, true, FENCE_TIMEOUT_NS)
                .map_err(|e| {
                    format!(
                        "vkWaitForFences failed after {} ms: {e} -- the GPU or its driver is \
                         wedged",
                        FENCE_TIMEOUT_NS / 1_000_000
                    )
                })?;
            self.device
                .reset_fences(&fences)
                .map_err(|e| format!("vkResetFences failed: {e}"))?;

            let readout: GpuReadout = self.buffers[binding::READOUT as usize].read_one();
            let active = readout.active_count as f32 / self.gc_dim.max(1) as f32;
            Ok((readout.ff, active))
        }
    }

    /// Copies the Purkinje weights out for saving. Only safe between steps, which is where the
    /// caller does it (at shutdown).
    pub fn read_weights(&self) -> Vec<f32> {
        // SAFETY: host-coherent mapping, no submission in flight.
        unsafe { self.buffers[binding::WEIGHTS as usize].read_vec(NUM_OUTPUTS * self.gc_dim) }
    }

    /// Installs previously-saved Purkinje weights.
    pub fn write_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let expected = NUM_OUTPUTS * self.gc_dim;
        if weights.len() != expected {
            return Err(format!(
                "weight vector has {} entries, this network needs {expected}",
                weights.len()
            ));
        }
        // SAFETY: as above.
        unsafe {
            self.buffers[binding::WEIGHTS as usize].write_slice(weights);
        }
        Ok(())
    }
}

impl Drop for GpuNet {
    fn drop(&mut self) {
        // SAFETY: teardown order is the reverse of creation, and `device_wait_idle` guarantees no
        // submission is still referencing any of it.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            for &m in &self.modules {
                self.device.destroy_shader_module(m, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            for b in &self.buffers {
                self.device.unmap_memory(b.memory);
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Byte size of each binding's buffer, indexed by binding number.
fn buffer_sizes(gc_dim: usize) -> [u64; binding::COUNT] {
    let f = std::mem::size_of::<f32>() as u64;
    let gc = gc_dim as u64;
    let mut sizes = [0u64; binding::COUNT];
    sizes[binding::PARAMS as usize] = std::mem::size_of::<GpuParams>() as u64;
    sizes[binding::MOSSY as usize] = MF_DIM as u64 * f;
    sizes[binding::GC_IDX as usize] = gc * GC_FAN_IN as u64 * 4;
    sizes[binding::GC_WEIGHT as usize] = gc * GC_FAN_IN as u64 * f;
    sizes[binding::GC_BIAS as usize] = gc * f;
    sizes[binding::GC_ACT as usize] = gc * f;
    sizes[binding::TRACE as usize] = gc * f;
    sizes[binding::WEIGHTS as usize] = NUM_OUTPUTS as u64 * gc * f;
    sizes[binding::READOUT as usize] = std::mem::size_of::<GpuReadout>() as u64;
    sizes
}

/// Picks a physical device with a compute queue, preferring an integrated GPU.
///
/// "Preferring integrated" is the unusual choice and it is deliberate: this network is tiny and
/// stepped at kilohertz, so what matters is submission latency and zero-copy access to host
/// memory, not throughput. On a laptop with a discrete GPU as well, the iGPU is both faster to
/// reach and the one that is not busy drawing.
///
/// Software rasterisers (lavapipe) are accepted only as a last resort and announced, because
/// "Vulkan is working" and "Vulkan is emulating a GPU on the CPU cores the control loop needs" are
/// very different situations to be in.
unsafe fn select_device(
    instance: &Instance,
    prefer_integrated: bool,
) -> Result<(vk::PhysicalDevice, u32, String), String> {
    let devices = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("vkEnumeratePhysicalDevices failed: {e}"))?;
    if devices.is_empty() {
        return Err("no Vulkan physical devices found".to_string());
    }

    let mut best: Option<(i32, vk::PhysicalDevice, u32, String)> = None;
    for &pd in &devices {
        let props = instance.get_physical_device_properties(pd);
        let families = instance.get_physical_device_queue_family_properties(pd);
        let Some(family) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.queue_count > 0)
        else {
            continue;
        };
        let name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let score = match props.device_type {
            vk::PhysicalDeviceType::INTEGRATED_GPU if prefer_integrated => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::DISCRETE_GPU if prefer_integrated => 2,
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::CPU => 0,
            _ => 1,
        };
        if best.as_ref().is_none_or(|(s, ..)| score > *s) {
            best = Some((score, pd, family as u32, name));
        }
    }

    let Some((score, pd, family, name)) = best else {
        return Err("no Vulkan device exposes a compute queue".to_string());
    };
    if score == 0 {
        log::warn!(
            "cerebellum: the only Vulkan device is '{name}', a software rasteriser -- it runs on \
             the same CPU cores as everything else, so expect it to compete with the control loop \
             rather than offload it"
        );
    }
    Ok((pd, family, name))
}

/// Allocates a storage buffer in memory that is simultaneously device-local and host-visible, and
/// leaves it mapped for the lifetime of the process.
///
/// The `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` combination is what removes staging buffers
/// from this design (see the module docs). `HOST_COHERENT` in particular means no explicit
/// `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges` around every step -- a
/// non-coherent type would work too but would put two more calls on the hot path, so it is only
/// taken as a fallback and never silently: the caller logs which one it got.
unsafe fn create_mapped_buffer(
    device: &Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<MappedBuffer, String> {
    let size = size.max(4); // Vulkan rejects zero-sized buffers
    let buffer = device
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .map_err(|e| format!("vkCreateBuffer({size} bytes) failed: {e}"))?;

    let reqs = device.get_buffer_memory_requirements(buffer);
    let ideal = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    let fallback = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = find_memory_type(mem_props, reqs.memory_type_bits, ideal)
        .or_else(|| find_memory_type(mem_props, reqs.memory_type_bits, fallback))
        .ok_or_else(|| {
            device.destroy_buffer(buffer, None);
            "no host-visible coherent memory type is available for a storage buffer".to_string()
        })?;

    let memory = device
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(type_index),
            None,
        )
        .map_err(|e| {
            device.destroy_buffer(buffer, None);
            format!("vkAllocateMemory({} bytes) failed: {e}", reqs.size)
        })?;
    device
        .bind_buffer_memory(buffer, memory, 0)
        .map_err(|e| format!("vkBindBufferMemory failed: {e}"))?;
    let ptr = device
        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        .map_err(|e| format!("vkMapMemory failed: {e}"))? as *mut u8;

    Ok(MappedBuffer {
        buffer,
        memory,
        ptr,
        size,
    })
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    allowed: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        allowed & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}
