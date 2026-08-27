//! Library crate backing the `so101_impedance_ctrl` binary, split out so unit tests can exercise
//! the protocol/shared-memory/control-law logic without a real serial port or RT scheduler.

pub mod cerebellum;
pub mod control;
pub mod feetech;
pub mod leader;
pub mod rt;
pub mod shm;
