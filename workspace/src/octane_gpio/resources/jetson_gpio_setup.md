# Jetson AGX Orin GPIO Pin Setup

## Problem

On the Jetson AGX Orin Developer Kit, most 40-pin header pins are **not enabled as GPIO by
default**. Calling `GPIO.setup()` and `GPIO.output()` will succeed without errors but produce
no voltage on the physical pin. Only pins explicitly configured as GPIO in the device tree will
actually drive the pad.

You can confirm a pin's current function:
```bash
sudo python3 /opt/nvidia/jetson-io/config-by-pin.py -p <pin>
# shows: gpio | unused | uarta | pwm8 | etc.
```

## OCTANE Pin Assignment

| Board Pin | Function   | Node constant  | Connected to        |
|-----------|------------|----------------|---------------------|
| 11        | ARM UP     | `PIN_ARM_UP`   | Relay 1 IN1         |
| 13        | ARM DOWN   | `PIN_ARM_DOWN` | Relay 1 IN2         |
| 15        | BUCKET A   | `PIN_BUCKET_A` | Relay 2 IN3         |
| 18        | BUCKET B   | `PIN_BUCKET_B` | Relay 2 IN4         |
| 6         | GND        | —              | Shared relay DC-    |

Pins 11 and 13 are configured as GPIO in the **base device tree** (no extra steps needed).
Pins 15 and 18 required a device tree overlay — see below.

## Enabling Additional Pins as GPIO

Pins 15 and 18 were enabled by running the following once on the Jetson (requires reboot):

```bash
sudo python3 - << 'EOF'
import sys
sys.path.insert(0, '/opt/nvidia/jetson-io')
from Jetson import board, io

b = board.Board()
b.set_active_header('Jetson 40pin Header')
h = b.header

for pin in [15, 18]:
    h.pin_set_state(pin, io.PinMode.GPIO, 'gp')
    print(f'pin {pin}: {h.pin_get_label(pin)}')

dtbo = b.create_dtbo_for_header()
msgs = b.configure_overlays([dtbo])
for m in msgs:
    print(m)
EOF

sudo reboot
```

This writes `/boot/jetson-io-hdr40-user-custom.dtbo` and adds a `JetsonIO` entry to
`/boot/extlinux/extlinux.conf` that loads the overlay at boot.

After reboot, verify:
```bash
sudo python3 /opt/nvidia/jetson-io/config-by-pin.py -p 15   # should print: gpio
sudo python3 /opt/nvidia/jetson-io/config-by-pin.py -p 18   # should print: gpio
```

## Adding More GPIO Pins in the Future

To enable additional pins, add their board pin numbers to the `for pin in [...]` list above
and re-run the script. The overlay is regenerated and replaces the previous one.

Available GPIO-capable pins (confirmed on this board, `tegra234-gpio`, no AON):
- Pin 7  — MCLK05
- Pin 12 — I2S2_CLK
- Pin 15 — GPIO27 ✓ (enabled)
- Pin 18 — GPIO35 ✓ (enabled)
- Pin 22 — GPIO17
- Pin 36 — UART1_CTS

## Test Script

`/media/csulunabotics/SSD2/OCTANE/test_relay.py` pulses all four relay pins in sequence.
Run it (stop ROS first) to verify hardware before launching the full system:
```bash
python3 /media/csulunabotics/SSD2/OCTANE/test_relay.py
```
