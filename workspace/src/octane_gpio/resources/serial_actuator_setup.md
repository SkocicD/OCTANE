# OCTANE Serial Actuator Setup

The Jetson sends a single byte to an Arduino over UART whenever the relay
state changes. The Arduino owns all four relay outputs.

## Wire-up

| Jetson 40-pin pin | Signal      | Arduino pin |
|-------------------|-------------|-------------|
| 8  (UART1_TXD)    | TX          | RX0 (D0)    |
| 6  (GND)          | GND         | GND         |

Only TX is used (Jetson talks, Arduino listens). Power the Arduino over USB
or its own supply — do not back-feed the Jetson.

UART device on the Jetson side: **`/dev/ttyTHS1`** (UART1, MMIO 0x3100000).
Confirm with `dmesg | grep ttyTHS`.

## Wire protocol

One byte per state change. High nibble always 0; low nibble is the relay state.

```
bit 3 (0x08) : arm UP        bit 1 (0x02) : bucket A
bit 2 (0x04) : arm DOWN      bit 0 (0x01) : bucket B
```

| arm | bucket | byte |
|-----|--------|------|
|  0  |   0    | 0x00 |
|  1  |   0    | 0x08 |
| -1  |   0    | 0x04 |
|  0  |   1    | 0x02 |
|  0  |  -1    | 0x01 |
|  1  |   1    | 0x0A |
|  1  |  -1    | 0x09 |
| -1  |   1    | 0x06 |
| -1  |  -1    | 0x05 |

Bytes with both bits of a pair set (0x0C, 0x03, 0x0F) must never appear — the
Jetson node forces them to 0 for that pair.

## Running

```
ros2 run octane_gpio serial_actuator_node
# or with a different port:
ros2 run octane_gpio serial_actuator_node --ros-args -p port:=/dev/ttyUSB0
```

Default baud 9600. Override with `-p baud:=<n>`.

## Arduino reference sketch

See `arduino_actuator.ino` in this directory. It:
- listens at 9600 baud
- maps each bit to a fixed digital output
- drops all relays after 500 ms with no traffic (watchdog)
