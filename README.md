# asfk — AFSK modem (transmitter)

Encodes text as an audio tone (AFSK: 1200Hz = bit 0, 2200Hz = bit 1, 100 baud) and plays it out the speaker. Pair with the [asfk Android app](../AndroidStudioProjects/asfk) as the receiver — it listens on the phone mic and decodes the tones back to text.

## Run

```bash
uv run main.py --text "hi"
```

Also plays back its own recording and self-decodes (loopback sanity check — always works, same clock, no acoustic path).

## Flags

- `--text "..."` — message to send (max 255 UTF-8 bytes)
- `--baud N` — bit rate (default 100; lower = slower but more robust over air)
- `--list-devices` / `--device N` — pick an audio output device

## Frame format

`preamble(16 alternating bits)` + `sync(0x7E7E)` + `length(1 byte)` + `payload` + `checksum(1 byte)`
