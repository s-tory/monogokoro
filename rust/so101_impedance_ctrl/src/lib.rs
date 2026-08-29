//! Library crate backing the `so101_impedance_ctrl` binary, split out so unit tests can exercise
//! the protocol/shared-memory/control-law logic without a real serial port or RT scheduler.

pub mod cerebellum;
pub mod control;
pub mod feetech;
pub mod leader;
/// The pontine relay. A sibling of `cerebellum`, not a part of it: the pontine nuclei sit in the
/// brainstem and relay cortex to the cerebellum, which is exactly this module's position in the
/// data flow -- between shared memory and the mossy fibres.
pub mod pontine;
pub mod rt;
pub mod shm;
