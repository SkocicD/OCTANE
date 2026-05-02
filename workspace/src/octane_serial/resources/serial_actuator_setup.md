# OCTANE Serial Actuator Setup

The Jetson sends a single byte to the Arduino over **USB** whenever the relay
state changes. The Arduino owns all four relay outputs.

USB is used (not the 40-pin header UART) because UART1 (`/dev/ttyTHS1`) on the
AGX Orin Devkit is silent on the header pads despite a correct pinmux — see
NVIDIA forum thread 362266; the fix is at the bootloader BCT level. USB sidesteps
that entirely.

## Wire-up

| From | To | Purpose |
|------|-----|---------|
| Jetson USB-A | Arduino USB-B (or Micro-USB) | data + power for the Arduino |

That's it. No 40-pin header wires. The relay module's GND should still tie to
the Arduino GND, and its VCC to the Arduino 5 V (or external 5 V if the
relays draw too much).

## Stable port path

The Arduino is referenced by its USB serial-by-id symlink so the device path
never changes across reboots, USB port swaps, or kernel updates:

```
/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_750313034313514022F1-if00
```

That string includes the Arduino's factory-programmed iSerial
(`750313034313514022F1`). If the Arduino is replaced with a different unit, the
serial number changes and the symlink path will need updating in two places:
the `port` default in `serial_actuator_node.py` and the `PORT` constant in
`serial_loopback.py` at the workspace root.

## Wire protocol

One byte per state change. Four data bits in the **low nibble** (LSB on the
right); high nibble is always zero (padding):

```
bit 7..4  always 0
bit 3  arm UP          bit 1  bucket A
bit 2  arm DOWN        bit 0  bucket B
```

So the wire byte is `0000ABCD` where `A B C D` are the four relay bits.

| arm | bucket | byte (binary) |
|-----|--------|---------------|
|  0  |   0    | `00000000`    |
|  1  |   0    | `00001000`    |
| -1  |   0    | `00000100`    |
|  0  |   1    | `00000010`    |
|  0  |  -1    | `00000001`    |
|  1  |   1    | `00001010`    |
|  1  |  -1    | `00001001`    |
| -1  |   1    | `00000110`    |
| -1  |  -1    | `00000101`    |

Bytes with both bits of a pair set (`0000 11xx`, `0000 xx11`, `0000 1111`) must
never appear — the Jetson node forces a pair to 00 if both are set.

## Running

```
ros2 run octane_gpio serial_actuator_node
# or override the port (e.g. for a different Arduino):
ros2 run octane_gpio serial_actuator_node \
    --ros-args -p port:=/dev/ttyACM0
```

Default baud 9600. Override with `-p baud:=<n>`.

## Arduino side

The sketch reads a single byte and uses the low 4 bits as the relay state:

```cpp
if (Serial.available()) {
  uint8_t b = Serial.read();
  bool arm_up    = b & 0b00001000;
  bool arm_down  = b & 0b00000100;
  bool bucket_a  = b & 0b00000010;
  bool bucket_b  = b & 0b00000001;
  // ... drive relay pins from those four booleans
}
```

A 500 ms watchdog (drop all relays if no new byte received) is recommended so
that a Jetson crash or unplug leaves the actuators safe.
