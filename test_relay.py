#!/usr/bin/env python3
"""Quick relay test — no ROS. Pulses each pin HIGH for 2 s.
Run: python3 test_relay.py
"""
import time
import Jetson.GPIO as GPIO

PINS = {
    11: 'ARM UP   (relay 1, IN1)',
    13: 'ARM DOWN (relay 1, IN2)',
    15: 'BUCKET A (relay 2, IN3)',
    18: 'BUCKET B (relay 2, IN4)',
}

GPIO.setmode(GPIO.BOARD)
for pin, label in PINS.items():
    try:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        print(f'  setup OK : pin {pin:2d} — {label}')
    except Exception as e:
        print(f'  setup ERR: pin {pin:2d} — {label}  → {e}')

print()
for pin, label in PINS.items():
    print(f'  >>> PIN {pin:2d} HIGH ({label})')
    try:
        GPIO.output(pin, GPIO.HIGH)
    except Exception as e:
        print(f'      ERROR: {e}')
    time.sleep(2.0)
    GPIO.output(pin, GPIO.LOW)
    print(f'      PIN {pin:2d} LOW')
    time.sleep(0.3)

GPIO.cleanup()
print('\nDone.')
