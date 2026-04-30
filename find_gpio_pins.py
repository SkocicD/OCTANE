#!/usr/bin/env python3
"""Pulses every candidate GPIO pin once per pass.

Run, probe pins with your relay wire, note which number is shown when it clicks.
Press Enter to run another pass. Ctrl+C to quit.
"""

import time
import Jetson.GPIO as GPIO

CANDIDATES = [7, 12, 15, 18, 22, 29, 31, 32, 33, 35, 36, 37, 38, 40]

GPIO.setmode(GPIO.BOARD)
setup_ok = []
for pin in CANDIDATES:
    try:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        setup_ok.append(pin)
    except Exception as e:
        print(f'skip pin {pin}: {e}')

print(f'Pins to test: {setup_ok}')
print('Probe header pins with relay wire — note the number shown when relay clicks.')
print('Press Enter to run a pass, Ctrl+C to quit.\n')

try:
    while True:
        input('Press Enter to run pass...')
        for pin in setup_ok:
            print(f'  PIN {pin:2d}', end='', flush=True)
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.15)
            print()
        print('-- pass done --\n')
except KeyboardInterrupt:
    pass

GPIO.cleanup()
print('Done.')
