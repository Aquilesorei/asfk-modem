#!/usr/bin/env python3
"""
afsk_modem.py — Minimal acoustic AFSK modem (loopback speaker -> mic)

Encodes text into bits, modulates each bit as an audio tone (2-frequency
FSK), plays the sound AND records at the same time (full-duplex), then
demodulates the recorded signal to recover the original text.

Dependencies:
    sudo dnf install portaudio portaudio-devel
    python3 -m venv venv && source venv/bin/activate
    pip install numpy sounddevice

Usage:
    python3 afsk_modem.py --text "Hello from my laptop"
    python3 afsk_modem.py --list-devices
"""

import argparse
import sys
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Missing 'sounddevice'. Install it with:")
    print("  pip install sounddevice   (and 'sudo dnf install portaudio' on PortAudio errors)")
    sys.exit(1)

SAMPLE_RATE = 44100
FREQ_0 = 1200.0     # bit 0 (like Bell 202 / AFSK1200)
FREQ_1 = 2200.0     # bit 1
SYNC_WORD = 0x7E7E   # transition-rich pattern, easy to detect
PREAMBLE_BITS = [i % 2 for i in range(16)]  # 0101...0101, AGC/clock lock-in


# ---------- Bit <-> byte utilities ----------

def byte_to_bits(b):
    return [(b >> i) & 1 for i in range(7, -1, -1)]  # MSB first


def bits_to_byte(bits):
    v = 0
    for bit in bits:
        v = (v << 1) | bit
    return v


def build_frame_bits(payload: bytes):
    bits = list(PREAMBLE_BITS)
    for b in [(SYNC_WORD >> 8) & 0xFF, SYNC_WORD & 0xFF]:
        bits += byte_to_bits(b)
    bits += byte_to_bits(len(payload))
    for byte in payload:
        bits += byte_to_bits(byte)
    checksum = sum(payload) % 256
    bits += byte_to_bits(checksum)
    return bits


# ---------- Modulation (TX) ----------

def modulate(bits, baud, sample_rate=SAMPLE_RATE):
    samples_per_bit = int(sample_rate / baud)
    signal = np.zeros(len(bits) * samples_per_bit, dtype=np.float64)
    phase = 0.0
    for i, bit in enumerate(bits):
        freq = FREQ_1 if bit else FREQ_0
        t = np.arange(samples_per_bit) / sample_rate
        # continuous phase across symbols -> no audible clicks
        seg = np.sin(2 * np.pi * freq * t + phase)
        signal[i * samples_per_bit:(i + 1) * samples_per_bit] = seg
        phase = (phase + 2 * np.pi * freq * samples_per_bit / sample_rate) % (2 * np.pi)

    # short (1ms) fade in/out to avoid clicks without weakening the last bit
    # on an acoustic air link (vs local loopback) where every bit matters
    # for sync/checksum.
    fade = int(0.001 * sample_rate)
    if fade > 0 and len(signal) > 2 * fade:
        ramp = np.linspace(0, 1, fade)
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal.astype(np.float32), samples_per_bit


# ---------- Demodulation (RX) ----------

def goertzel_power(samples, freq, sample_rate):
    n = len(samples)
    k = int(0.5 + n * freq / sample_rate)
    w = 2 * np.pi * k / n
    coeff = 2 * np.cos(w)
    q0 = q1 = q2 = 0.0
    for s in samples:
        q0 = coeff * q1 - q2 + s
        q2 = q1
        q1 = q0
    real = q1 - q2 * np.cos(w)
    imag = q2 * np.sin(w)
    return real * real + imag * imag


def decode_bit(window, sample_rate):
    p0 = goertzel_power(window, FREQ_0, sample_rate)
    p1 = goertzel_power(window, FREQ_1, sample_rate)
    return 1 if p1 > p0 else 0


def find_sync_offset(recorded, samples_per_bit, sample_rate, search_seconds=1.5):
    expected = list(PREAMBLE_BITS)
    for b in [(SYNC_WORD >> 8) & 0xFF, SYNC_WORD & 0xFF]:
        expected += byte_to_bits(b)
    n_bits = len(expected)

    max_search = min(len(recorded) - n_bits * samples_per_bit,
                      int(search_seconds * sample_rate))
    if max_search <= 0:
        return None, None

    step = max(1, samples_per_bit // 8)
    best_offset, best_errors = None, n_bits + 1

    for offset in range(0, max_search, step):
        errors = 0
        for i, expected_bit in enumerate(expected):
            start = offset + i * samples_per_bit
            window = recorded[start:start + samples_per_bit]
            if len(window) < samples_per_bit:
                errors = n_bits + 1
                break
            if decode_bit(window, sample_rate) != expected_bit:
                errors += 1
                if errors >= best_errors:
                    break
        if errors < best_errors:
            best_errors, best_offset = errors, offset
            if errors == 0:
                break

    return best_offset, best_errors, n_bits


def demodulate(recorded, samples_per_bit, sample_rate):
    result = find_sync_offset(recorded, samples_per_bit, sample_rate)
    offset, errors, header_bits = result
    if offset is None or errors > header_bits * 0.25:
        return None, f"Sync not found (best error: {errors}/{header_bits} bits)"

    pos = offset + header_bits * samples_per_bit

    def read_byte():
        nonlocal pos
        bits = []
        for _ in range(8):
            window = recorded[pos:pos + samples_per_bit]
            if len(window) < samples_per_bit:
                return None
            bits.append(decode_bit(window, sample_rate))
            pos += samples_per_bit
        return bits_to_byte(bits)

    length = read_byte()
    if length is None:
        return None, "Signal too short to read the length"
    # length 0 is rejected outright: an empty payload's checksum is always 0, so a
    # garbled decode that happens to misread both bytes as 0 would otherwise pass
    # as a false-positive "empty message".
    if length == 0:
        return None, "Invalid length (0)"

    payload = bytearray()
    for _ in range(length):
        b = read_byte()
        if b is None:
            return None, "Signal too short to read the payload"
        payload.append(b)

    checksum = read_byte()
    if checksum is None:
        return None, "Signal too short to read the checksum"

    if sum(payload) % 256 != checksum:
        return None, f"Invalid checksum (payload likely corrupted, {errors} sync errors)"

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = repr(bytes(payload))

    return text, f"OK (offset={offset} samples, {errors}/{header_bits} sync errors)"


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Acoustic AFSK modem (loopback speaker -> mic)")
    parser.add_argument("--text", default="Je pense donc je suis", help="Text to transmit")
    parser.add_argument("--baud", type=int, default=100, help="Baud rate (default: 100)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Index of the audio device to use")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    payload = args.text.encode("utf-8")
    if len(payload) > 255:
        print("Text too long (max 255 bytes in UTF-8)")
        sys.exit(1)

    bits = build_frame_bits(payload)
    tx_signal, samples_per_bit = modulate(bits, args.baud)

    # trailing guard silence so the end of the signal is captured too
    guard = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    tx_padded = np.concatenate([tx_signal, guard])

    duration_s = len(tx_padded) / SAMPLE_RATE
    print(f"Transmitting: \"{args.text}\" ({len(payload)} bytes, {args.baud} baud, {duration_s:.1f}s)")
    print(f"Frequencies: bit0={FREQ_0}Hz  bit1={FREQ_1}Hz")

    if args.device is not None:
        sd.default.device = (args.device, args.device)

    recorded = sd.playrec(tx_padded, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    recorded = recorded.flatten()

    print("Decoding...")
    text, status = demodulate(recorded, samples_per_bit, SAMPLE_RATE)

    print(f"Status : {status}")
    if text is not None:
        print(f"Decoded: \"{text}\"")
        print("Success!" if text == args.text else "Decoded but different from the original text.")
    else:
        print("Decoding failed. Try increasing the speaker volume")
        print("or lowering --baud (e.g. --baud 50) for more robustness.")


if __name__ == "__main__":
    main()
