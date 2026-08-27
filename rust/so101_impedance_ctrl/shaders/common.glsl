// Shared bindings and constants for the cerebellum's three compute stages.
//
// Every stage uses the same single descriptor set, so one set is allocated and bound once at
// init and the per-tick cost is a buffer write plus a queue submit -- no descriptor churn.
//
// This file is the GPU half of a two-implementation design: `src/cerebellum/net.rs` holds the
// readable version of exactly this math and is the authority on what it should compute.
// `tests/cerebellum_net_tests.rs` runs both and compares. Change one, change the other.

// Must equal `net::GC_FAN_IN`. Checked at init (`gpu.rs` asserts against the Rust constant).
const uint GC_FAN_IN = 4u;
// Must equal `net::NUM_OUTPUTS` / `shm::NUM_MOTORS`.
const uint NUM_OUTPUTS = 6u;

layout(std430, set = 0, binding = 0) readonly buffer Params {
    uint  gc_dim;
    uint  learn;         // 0 disables the plasticity dispatch's effect
    float theta;         // Golgi inhibition threshold, driven by the host's sparsity integrator
    float trace_decay;   // per-tick eligibility-trace decay
    float rate;          // LMS/Hebbian step size
    float leak;          // heterosynaptic decay coefficient
    float cf[NUM_OUTPUTS]; // climbing fibre: the reflex's standing duty, already gated
};

layout(std430, set = 0, binding = 1) readonly  buffer MossyFibres { float mf[]; };
layout(std430, set = 0, binding = 2) readonly  buffer GranuleIdx  { uint  gc_idx[]; };
layout(std430, set = 0, binding = 3) readonly  buffer GranuleW    { float gc_w[]; };
layout(std430, set = 0, binding = 4) readonly  buffer GranuleBias { float gc_bias[]; };
layout(std430, set = 0, binding = 5)           buffer GranuleAct  { float gc[]; };
layout(std430, set = 0, binding = 6)           buffer Eligibility { float trace[]; };
layout(std430, set = 0, binding = 7)           buffer Purkinje    { float weights[]; };
layout(std430, set = 0, binding = 8)           buffer Readout {
    float ff[NUM_OUTPUTS];
    uint  active_count;  // granule cells above threshold this tick; host-zeroed before submit
};
