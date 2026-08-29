//! Packet-framing round-trip tests -- no serial port or hardware needed.

use so101_impedance_ctrl::feetech::{
    build_packet, checksum, decode_sign_magnitude, encode_sign_magnitude, parse_status_packet,
    BROADCAST_ID, CURRENT_SIGN_BIT, INST_READ, INST_SYNC_READ,
};

#[test]
fn build_packet_frames_a_read_instruction() {
    // READ id=1, addr=56 (Present_Position), len=2 bytes.
    let packet = build_packet(1, INST_READ, &[56, 2]);
    assert_eq!(&packet[0..2], &[0xFF, 0xFF]);
    assert_eq!(packet[2], 1); // id
    assert_eq!(packet[3], 4); // len = instruction(1) + params(2) + checksum(1)
    assert_eq!(packet[4], INST_READ);
    assert_eq!(packet[5], 56);
    assert_eq!(packet[6], 2);
    let expected_checksum = checksum(&[1, 4, INST_READ, 56, 2]);
    assert_eq!(packet[7], expected_checksum);
    assert_eq!(packet.len(), 8);
}

#[test]
fn build_packet_frames_a_sync_read_instruction() {
    // SYNC_READ addr=56 (Present_Position), 2 bytes, for the SO101's 6 motors.
    let ids = [1u8, 2, 3, 4, 5, 6];
    let mut params = vec![56u8, 2];
    params.extend_from_slice(&ids);
    let packet = build_packet(BROADCAST_ID, INST_SYNC_READ, &params);

    assert_eq!(&packet[0..2], &[0xFF, 0xFF]);
    assert_eq!(packet[2], BROADCAST_ID);
    // Per protocol, LEN = n_ids + 4 (instruction + addr + data_len + checksum).
    assert_eq!(packet[3], ids.len() as u8 + 4);
    assert_eq!(packet[4], INST_SYNC_READ);
    assert_eq!(packet[5], 56); // addr
    assert_eq!(packet[6], 2); // bytes per motor
    assert_eq!(&packet[7..13], &ids);
    assert_eq!(packet.len(), 14);
}

/// A SYNC_READ reply is just N back-to-back status packets in request order; this is exactly what
/// `FeetechBus::sync_read` slices into fixed `6 + size` byte chunks and parses.
#[test]
fn sync_read_reply_parses_as_back_to_back_status_packets() {
    let ids = [1u8, 2, 3, 4, 5, 6];
    let positions: [u16; 6] = [2048, 100, 4095, 0, 1234, 3000];

    let mut stream = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let lo = (positions[i] & 0xFF) as u8;
        let hi = (positions[i] >> 8) as u8;
        let body = [id, 4, 0, lo, hi]; // id, len(=data 2 + error 1 + checksum 1), error, data
        stream.extend_from_slice(&[0xFF, 0xFF]);
        stream.extend_from_slice(&body);
        stream.push(checksum(&body));
    }

    let resp_len = 6 + 2; // header(2) + id + len + error + data(2) + checksum
    assert_eq!(stream.len(), resp_len * ids.len());

    for (i, &id) in ids.iter().enumerate() {
        let chunk = &stream[i * resp_len..(i + 1) * resp_len];
        let (resp_id, error, data) = parse_status_packet(chunk).expect("chunk must parse");
        assert_eq!(resp_id, id);
        assert_eq!(error, 0);
        let value = (data[0] as u16) | ((data[1] as u16) << 8);
        assert_eq!(value, positions[i]);
    }
}

#[test]
fn parse_status_packet_round_trips() {
    let body = [1u8, 4, 0, 10, 20]; // id, len, error, param_lo, param_hi
    let cksum = checksum(&body);
    let mut raw = vec![0xFF, 0xFF];
    raw.extend_from_slice(&body);
    raw.push(cksum);

    let (id, error, params) = parse_status_packet(&raw).expect("valid packet must parse");
    assert_eq!(id, 1);
    assert_eq!(error, 0);
    assert_eq!(params, vec![10, 20]);
}

#[test]
fn parse_status_packet_rejects_bad_checksum() {
    let mut raw = vec![0xFF, 0xFF, 1, 4, 0, 10, 20, 0x00];
    assert!(
        parse_status_packet(&raw).is_none(),
        "checksum 0x00 should be invalid here"
    );

    raw[7] = checksum(&[1, 4, 0, 10, 20]);
    assert!(parse_status_packet(&raw).is_some());
}

#[test]
fn parse_status_packet_rejects_bad_header() {
    let raw = vec![0x00, 0x00, 1, 4, 0, 10, 20, 0xFF];
    assert!(parse_status_packet(&raw).is_none());
}

#[test]
fn parse_status_packet_rejects_length_mismatch() {
    // Declares len=4 (5 bytes after header) but only supplies 3 -- must not panic/slice-oob.
    let raw = vec![0xFF, 0xFF, 1, 4, 0];
    assert!(parse_status_packet(&raw).is_none());
}

#[test]
fn sign_magnitude_round_trips_positive_and_negative() {
    for &(value, bit) in &[
        (0i32, 15u32),
        (100, 15),
        (-100, 15),
        (2047, 11),
        (-2047, 11),
        (1, 11),
    ] {
        let encoded = encode_sign_magnitude(value, bit);
        let decoded = decode_sign_magnitude(encoded, bit);
        assert_eq!(decoded, value, "round trip failed for {value} at bit {bit}");
    }
}

#[test]
fn sign_magnitude_negative_zero_decodes_as_zero() {
    // Sign bit set but zero magnitude -- some firmwares emit this for "no direction" at rest.
    let raw = 1u16 << 11;
    assert_eq!(decode_sign_magnitude(raw, 11), 0);
}

/// The four `Present_Current` samples that exposed the raw-`u16` read, decoded at the bit the
/// register actually signs at.
///
/// Bit 10 is the plausible wrong answer -- it is what the neighbouring `Present_Load` uses -- and
/// it fails silently rather than loudly: it recovers the right magnitude from every one of these
/// and just drops the sign, so the joint's load reads positive while it pushes the other way.
#[test]
fn present_current_decodes_sign_magnitude_at_bit_15() {
    for &(raw, expected) in &[
        (0x800Au16, -10i32),
        (0x8016, -22),
        (0x8019, -25),
        (0x8045, -69),
        (0x0016, 22),
    ] {
        assert_eq!(decode_sign_magnitude(raw, CURRENT_SIGN_BIT), expected);
        assert_eq!(
            decode_sign_magnitude(raw, 10),
            expected.abs(),
            "bit 10 loses the sign"
        );
    }
}

/// A stray byte before a reply must not blame the wrong servo.
///
/// The first parser sliced the response into fixed `6 + size` chunks, so one extra byte anywhere
/// shifted every later chunk and surfaced as "bad packet for motor <last id>" -- pointing at a
/// perfectly healthy servo. Header scanning resynchronises instead.
#[test]
fn sync_read_reply_survives_a_stray_byte_between_replies() {
    let ids = [1u8, 2, 3];
    let positions: [u16; 3] = [1000, 2000, 3000];

    let mut stream = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if i == 1 {
            stream.push(0x00); // line noise between replies
        }
        let lo = (positions[i] & 0xFF) as u8;
        let hi = (positions[i] >> 8) as u8;
        let body = [id, 4, 0, lo, hi];
        stream.extend_from_slice(&[0xFF, 0xFF]);
        stream.extend_from_slice(&body);
        stream.push(checksum(&body));
    }

    // Walk it the way `sync_read` does: find each header, then parse a fixed-size packet there.
    let resp_len = 6 + 2;
    let mut cursor = 0usize;
    for (i, &id) in ids.iter().enumerate() {
        let off = stream[cursor..]
            .windows(2)
            .position(|w| w == [0xFF, 0xFF])
            .expect("header must be findable");
        let start = cursor + off;
        let (resp_id, _, data) =
            parse_status_packet(&stream[start..start + resp_len]).expect("packet must parse");
        assert_eq!(resp_id, id);
        assert_eq!((data[0] as u16) | ((data[1] as u16) << 8), positions[i]);
        cursor = start + resp_len;
    }
}
