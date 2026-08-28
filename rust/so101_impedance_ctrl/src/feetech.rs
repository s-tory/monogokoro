//! Feetech STS/SMS-series servo protocol framing, hand-implemented for RT-loop use.
//!
//! This repo's Python `FeetechMotorsBus` (`src/lerobot/motors/feetech/feetech.py`) delegates the
//! actual wire protocol to the external `feetech-servo-sdk` package, whose source isn't vendored
//! here -- unsuitable for an RT loop anyway, since shelling out to a Python process per tick would
//! reintroduce interpreter/GC/scheduling jitter. This module reimplements the framing natively.
//!
//! Register addresses are mirrored from `src/lerobot/motors/feetech/tables.py`
//! (`STS_SMS_SERIES_CONTROL_TABLE`) -- keep the two in sync if either changes.

use std::io::{Read, Write};
use std::time::Duration;

use serialport::SerialPort;

pub const HEADER: [u8; 2] = [0xFF, 0xFF];
pub const BROADCAST_ID: u8 = 0xFE;

pub const INST_PING: u8 = 0x01;
pub const INST_READ: u8 = 0x02;
pub const INST_WRITE: u8 = 0x03;
/// Only valid on protocol-0 parts (STS/SMS series, which includes the SO101's sts3215). The
/// protocol-1 SCS series has no SYNC_READ -- mirrors the guard in
/// `FeetechMotorsBus._assert_protocol_is_compatible` on the Python side.
pub const INST_SYNC_READ: u8 = 0x82;
pub const INST_SYNC_WRITE: u8 = 0x83;

// Register map: (address, size_bytes), mirrored from
// lerobot/motors/feetech/tables.py::STS_SMS_SERIES_CONTROL_TABLE.
/// **EPROM.** Delay before a servo answers, in 2 us units. Ships at 250 (= 500 us), which is
/// brutal when six servos answer one SYNC_READ in sequence. `FeetechMotorsBus.configure_motors`
/// zeroes it on the Python side; this daemon does the same at startup.
pub const REG_RETURN_DELAY_TIME: (u8, u8) = (7, 1);
/// **EPROM.** Bit 4 (0x10) selects the angle feedback mode. It must be *clear* on the sts3215 so
/// `Present_Position` stays in `[0, resolution-1]` instead of overflowing or going negative --
/// same fixup `configure_motors` applies in Python. A wrapped position would corrupt both the
/// impedance error term and the finite-differenced velocity.
pub const REG_PHASE: (u8, u8) = (18, 1);
/// Bit within [`REG_PHASE`] that must be cleared.
pub const PHASE_ANGLE_FEEDBACK_BIT: u32 = 0x10;
pub const REG_P_COEFFICIENT: (u8, u8) = (21, 1);
pub const REG_D_COEFFICIENT: (u8, u8) = (22, 1);
pub const REG_I_COEFFICIENT: (u8, u8) = (23, 1);
/// **EPROM.** Shifts the reported `Present_Position` so a joint's working range does not straddle
/// the 4095/0 encoder wrap. Sign-magnitude with the sign at bit 11 (see [`HOMING_OFFSET_SIGN_BIT`]),
/// *not* bit 15 like `Goal_Position`.
pub const REG_HOMING_OFFSET: (u8, u8) = (31, 2);
/// Sign bit for [`REG_HOMING_OFFSET`], mirroring `STS_SMS_SERIES_ENCODINGS_TABLE["Homing_Offset"]`.
pub const HOMING_OFFSET_SIGN_BIT: u32 = 11;
/// **EPROM.** Position limits the servo enforces internally; written from the calibration
/// alongside the homing offset, matching `FeetechMotorsBus.write_calibration`.
pub const REG_MIN_POSITION_LIMIT: (u8, u8) = (9, 2);
pub const REG_MAX_POSITION_LIMIT: (u8, u8) = (11, 2);
/// **EPROM register.** Everything below address 40 lives in EPROM and is write-protected while
/// `REG_LOCK` is 1 -- see `write_operating_mode`, which does the required unlock dance.
pub const REG_OPERATING_MODE: (u8, u8) = (33, 1);

/// `Operating_Mode` value for open-loop PWM control, mirroring `OperatingMode.PWM` in
/// `src/lerobot/motors/feetech/feetech.py`. This is EPROM: writing it needs the torque-off +
/// unlock dance in `control.rs`, which is why nothing writes it directly.
pub const OPERATING_MODE_PWM: u32 = 2;
pub const REG_TORQUE_ENABLE: (u8, u8) = (40, 1);
/// EPROM write-protect latch: 0 = EPROM writable, 1 = locked. Mirrors the `Lock` writes in
/// `FeetechMotorsBus.disable_torque`/`enable_torque` on the Python side.
pub const REG_LOCK: (u8, u8) = (55, 1);
pub const REG_GOAL_POSITION: (u8, u8) = (42, 2);
/// Overloaded as the PWM duty-cycle command register when `Operating_Mode = PWM` (addr 0x2c).
/// tables.py calls this register `Goal_Time` for POSITION mode; per the Feetech datasheet, in PWM
/// mode bit 11 (not bit 15, unlike `Goal_Position`) is the sign/direction bit.
pub const REG_GOAL_PWM: (u8, u8) = (44, 2);
pub const REG_PRESENT_POSITION: (u8, u8) = (56, 2);
pub const REG_PRESENT_CURRENT: (u8, u8) = (69, 2);
/// Supply voltage as the servo sees it, in units of 0.1 V per the datasheet. Read once at startup
/// rather than in the loop: it costs a transaction, and the control law has no use for it.
pub const REG_PRESENT_VOLTAGE: (u8, u8) = (62, 1);
/// Case temperature in degrees C.
pub const REG_PRESENT_TEMPERATURE: (u8, u8) = (63, 1);

/// Bit index of the direction flag in the PWM command register (see [`REG_GOAL_PWM`]).
///
/// **Measured as bit 10, not the bit 11 that `feetech.py`'s `OperatingMode` docstring states.**
/// With bit 11, flipping the flag did not reverse the joint -- it drove it harder the same way
/// (-227 -> -411 ticks for an identical commanded magnitude), i.e. the bit was being consumed as
/// extra magnitude and clamped to full duty. With bit 10 the same probe reversed cleanly
/// (-227 -> +130). That fits a 10-bit duty field: 0-1023 is the smallest range that covers the
/// 0-1000 duty scale, leaving bit 10 as the sign.
pub const PWM_SIGN_BIT: u32 = 10;
/// Bit index of the sign bit for `Present_Position`/`Goal_Position` (full 16-bit range, bit 15).
pub const POSITION_SIGN_BIT: u32 = 15;

/// Bit index of the direction flag in `Present_Current` (addr 69).
///
/// **Measured as bit 15**, and it had to be measured: this is the one register the upstream
/// `STS_SMS_SERIES_ENCODINGS_TABLE` omits, and its neighbours disagree -- `Present_Load` signs at
/// bit 10 while `Present_Position` signs at bit 15, so neither is a safe guess.
///
/// Holding a gravity-loaded joint at a steady duty read `22` with the load one way and `0x8016`
/// with it the other: the same magnitude, with bit 15 set. Bits 11-14 stayed clear at every
/// magnitude observed (`0x800A`, `0x8016`, `0x8019`, `0x8045`), which is what rules bit 10 out --
/// decoding those with a bit-10 sign recovers the magnitude but silently drops the sign.
pub const CURRENT_SIGN_BIT: u32 = 15;

/// Sign-magnitude decode: `sign_bit` holds the sign (1 = negative), the lower bits hold the
/// magnitude. Mirrors `lerobot.motors.encoding_utils.decode_sign_magnitude`.
pub fn decode_sign_magnitude(value: u16, sign_bit: u32) -> i32 {
    let sign_mask = 1u16 << sign_bit;
    let magnitude_mask = sign_mask - 1;
    let magnitude = (value & magnitude_mask) as i32;
    if value & sign_mask != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Sign-magnitude encode, inverse of [`decode_sign_magnitude`]. `value`'s magnitude is truncated
/// to fit in `sign_bit` bits if it overflows.
pub fn encode_sign_magnitude(value: i32, sign_bit: u32) -> u16 {
    let sign_mask = 1u16 << sign_bit;
    let magnitude_mask = (sign_mask - 1) as i32;
    let magnitude = value.unsigned_abs() as i32 & magnitude_mask;
    let mut out = magnitude as u16;
    if value < 0 {
        out |= sign_mask;
    }
    out
}

/// Feetech checksum: one's complement of the sum of all bytes from `id` through the last
/// parameter byte (i.e. everything in the packet except the two header bytes and the checksum
/// byte itself).
pub fn checksum(bytes: &[u8]) -> u8 {
    let sum: u32 = bytes.iter().map(|&b| b as u32).sum();
    (!sum) as u8
}

/// Builds a full instruction packet: `[0xFF, 0xFF, id, len, instruction, ...params, checksum]`.
pub fn build_packet(id: u8, instruction: u8, params: &[u8]) -> Vec<u8> {
    let len = (params.len() + 2) as u8; // instruction + params + checksum byte, per protocol
    let mut body = Vec::with_capacity(3 + params.len());
    body.push(id);
    body.push(len);
    body.push(instruction);
    body.extend_from_slice(params);
    let cksum = checksum(&body);

    let mut packet = Vec::with_capacity(2 + body.len() + 1);
    packet.extend_from_slice(&HEADER);
    packet.extend_from_slice(&body);
    packet.push(cksum);
    packet
}

/// Parses a status/response packet, returning `(id, error, params)`.
/// Returns `None` if the header, declared length, or checksum don't validate.
pub fn parse_status_packet(bytes: &[u8]) -> Option<(u8, u8, Vec<u8>)> {
    if bytes.len() < 6 || bytes[0] != HEADER[0] || bytes[1] != HEADER[1] {
        return None;
    }
    let id = bytes[2];
    let len = bytes[3] as usize;
    if bytes.len() != 4 + len {
        return None;
    }
    let error = bytes[4];
    let params = bytes[5..4 + len - 1].to_vec();
    let expected = checksum(&bytes[2..4 + len - 1]);
    if bytes[4 + len - 1] != expected {
        return None;
    }
    Some((id, error, params))
}

/// Byte offset of the next `0xFF 0xFF` packet header, if any.
fn find_header(bytes: &[u8]) -> Option<usize> {
    bytes.windows(2).position(|w| w == HEADER)
}

fn decode_unsigned(data: &[u8]) -> i32 {
    match data.len() {
        1 => data[0] as i32,
        2 => (data[0] as i32) | ((data[1] as i32) << 8),
        _ => 0,
    }
}

fn encode_value(value: u32, size: u8, params: &mut Vec<u8>) {
    match size {
        1 => params.push(value as u8),
        2 => {
            params.push((value & 0xFF) as u8);
            params.push(((value >> 8) & 0xFF) as u8);
        }
        _ => panic!("unsupported register size {size}"),
    }
}

/// A live connection to the SO101's single half-duplex UART, exclusively owned by this daemon
/// (see the plan's bus-ownership decision: no other process may touch this port while the daemon
/// runs, including for the gripper).
pub struct FeetechBus {
    port: Box<dyn SerialPort>,
}

impl FeetechBus {
    pub fn open(path: &str, baud: u32, timeout: Duration) -> std::io::Result<Self> {
        let port = serialport::new(path, baud).timeout(timeout).open()?;
        Ok(Self { port })
    }

    fn transact(&mut self, packet: &[u8], expected_len: usize) -> std::io::Result<Vec<u8>> {
        self.port.write_all(packet)?;
        self.port.flush()?;
        let mut buf = vec![0u8; expected_len];
        self.port.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Writes `value` to `reg` on servo `id` and waits for the status-packet ack.
    pub fn write_register(&mut self, id: u8, reg: (u8, u8), value: u32) -> std::io::Result<()> {
        let (addr, size) = reg;
        let mut params = vec![addr];
        encode_value(value, size, &mut params);
        let packet = build_packet(id, INST_WRITE, &params);
        // WRITE status packet is just [header(2), id, len=2, error, checksum] = 6 bytes.
        let resp = self.transact(&packet, 6)?;
        parse_status_packet(&resp).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "bad Feetech ack")
        })?;
        Ok(())
    }

    /// Reads `reg` from servo `id`, returning the raw unsigned register value.
    pub fn read_register(&mut self, id: u8, reg: (u8, u8)) -> std::io::Result<i32> {
        let (addr, size) = reg;
        let params = [addr, size];
        let packet = build_packet(id, INST_READ, &params);
        let resp = self.transact(&packet, 6 + size as usize)?;
        let (_, _, data) = parse_status_packet(&resp).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "bad Feetech status packet")
        })?;
        Ok(decode_unsigned(&data))
    }

    /// Reads the same register from several servos in **one** bus transaction, returning the raw
    /// values in the same order as `ids`.
    ///
    /// This is what makes a 1 kHz control loop feasible. Reading position + current for 6 motors
    /// with `read_register` costs 12 round-trips: at 1 Mbaud each is 16 bytes ~= 160 us, so
    /// ~1.92 ms/tick -- already over a 1 ms budget before any PWM write. SYNC_READ collapses each
    /// register into a single request plus back-to-back replies (~0.6 ms for both registers).
    ///
    /// Trade-off: one unresponsive servo fails the whole batch (the fixed-length read times out)
    /// rather than just its own slot, so the caller loses the entire tick's data instead of one
    /// motor's. The control loop handles that by holding its last known-good values.
    pub fn sync_read(&mut self, reg: (u8, u8), ids: &[u8]) -> std::io::Result<Vec<i32>> {
        let (addr, size) = reg;
        let mut params = Vec::with_capacity(2 + ids.len());
        params.push(addr);
        params.push(size);
        params.extend_from_slice(ids);
        let packet = build_packet(BROADCAST_ID, INST_SYNC_READ, &params);

        // Each servo answers with its own status packet:
        // [header(2), id, len, error, data(size), checksum] = 6 + size bytes, in request order.
        let resp_len = 6 + size as usize;
        let resp = self.transact(&packet, resp_len * ids.len())?;

        // Locate each reply by its 0xFF 0xFF header rather than assuming the stream is exactly
        // `resp_len`-aligned. A single stray byte -- line noise on the half-duplex bus, or a servo
        // that answered slightly differently -- would otherwise shift every subsequent chunk and
        // surface as a bogus "bad packet for motor <last id>", blaming the wrong servo.
        let mut values = Vec::with_capacity(ids.len());
        let mut cursor = 0usize;
        for &id in ids {
            let start = find_header(&resp[cursor..])
                .map(|off| cursor + off)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "SYNC_READ: no reply header found for motor {id} (got {} of {} \
                             expected bytes: {:02X?})",
                            resp.len().saturating_sub(cursor),
                            resp_len,
                            &resp[cursor.min(resp.len())..]
                        ),
                    )
                })?;
            let end = start + resp_len;
            if end > resp.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("SYNC_READ: reply for motor {id} is truncated"),
                ));
            }
            let (resp_id, _, data) = parse_status_packet(&resp[start..end]).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "bad Feetech SYNC_READ status packet for motor {id}: {:02X?}",
                        &resp[start..end]
                    ),
                )
            })?;
            if resp_id != id {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("SYNC_READ reply out of order: expected motor {id}, got {resp_id}"),
                ));
            }
            values.push(decode_unsigned(&data));
            cursor = end;
        }
        Ok(values)
    }

    /// Broadcasts a single register write to several servos in one bus transaction (no status
    /// response is returned by SYNC_WRITE, per protocol) -- used for the per-tick PWM commands to
    /// keep RT-loop bus time bounded and deterministic.
    pub fn sync_write(&mut self, reg: (u8, u8), values: &[(u8, u32)]) -> std::io::Result<()> {
        let (addr, size) = reg;
        let mut params = vec![addr, size];
        for &(id, value) in values {
            params.push(id);
            encode_value(value, size, &mut params);
        }
        let packet = build_packet(BROADCAST_ID, INST_SYNC_WRITE, &params);
        self.port.write_all(&packet)?;
        self.port.flush()?;
        Ok(())
    }
}
