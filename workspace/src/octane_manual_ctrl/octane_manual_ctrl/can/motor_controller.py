# Motor node IDs 1-6
# Left side:  1=front-left, 2=mid-left,  3=back-left
# Right side: 4=back-right, 5=mid-right, 6=front-right  (right side polarity flipped — mirrored mount)
# Node 3 (back-left) is additionally flipped relative to the other left-side motors

from octane_manual_ctrl.can.can_open_handler import CANOpenMessageFactory

MOTOR_MODE_INDEX    = b'\x61\x00'
MOTOR_MODE_SUBINDEX = b'\x02'

# mode byte: control=1, speed_reg=1, sensory=1, open_loop=1
MOTOR_MODE_VALUE = b'\x0f'

MIN_SPEED_RAW  = 100   # below this, motor stalls — send stop instead
MIN_SPEED_HYST = 25    # hysteresis: only stop when raw < (MIN_SPEED_RAW - HYST) while already running
MAX_SPEED_RAW  = 2000


class MotorController:

    def __init__(self, node_id: int, transceiver):
        self._node_id = node_id
        self._tx = transceiver
        self._factory = CANOpenMessageFactory()
        self._right_side = node_id > 3   # right-side motors (4,5,6) mount mirrored
        self._extra_flip = node_id == 3  # back-left physically opposite the other left-side motors
        self._running = False             # tracks whether motor is currently spinning

    def turn_on(self):
        can_id, data = self._factory.create_nmt_start(self._node_id)
        self._tx.send(can_id, data)

    def set_mode(self):
        can_id, data = self._factory.create_sdo_write(
            self._node_id, MOTOR_MODE_INDEX, MOTOR_MODE_SUBINDEX, MOTOR_MODE_VALUE)
        self._tx.send(can_id, data)

    def move(self, speed: float, reverse: bool = False):
        """Drive motor. speed in [0.0, 1.0]. reverse flips direction."""
        speed = max(0.0, min(1.0, speed))

        if self._right_side:
            reverse = not reverse
        if self._extra_flip:
            reverse = not reverse

        raw = int(MAX_SPEED_RAW * speed)
        # Hysteresis: stop threshold is lower while motor is already spinning
        # so we don't oscillate on/off at the minimum boundary.
        stop_threshold = (MIN_SPEED_RAW - MIN_SPEED_HYST) if self._running else MIN_SPEED_RAW
        if raw < stop_threshold:
            self.stop()
            return

        self._running = True
        ctrl = (0x01 | (0x02 if reverse else 0x00)).to_bytes(1, 'little')
        spd  = raw.to_bytes(2, 'little')
        can_id, data = self._factory.create_pdo_write(self._node_id, ctrl + spd)
        self._tx.send(can_id, data)

    def stop(self):
        self._running = False
        can_id, data = self._factory.create_pdo_write(self._node_id, b'')
        self._tx.send(can_id, data)
