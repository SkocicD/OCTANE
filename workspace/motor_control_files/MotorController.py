from CANTransceiver import CANTransceiver
from CANOpenHandler import CANOpenMessageFactory
import time

MOTOR_SPEED_INDEX = b'\x61\x0B'
MOTOR_SPEED_SUBINDEX = b'\x00'
MOTOR_CONTROL_INDEX = b'\x61\x00'
MOTOR_CONTROL_SUBINDEX = b'\x01'
MOTOR_MODE_INDEX = b'\x61\x00'
MOTOR_MODE_SUBINDEX = b'\x02'
HEARTBEAT_INDEX = b'\x10\x17'
HEARTBEAT_SUBINDEX = b'\x00'
NODE_ID_INDEX = b'\x10\x06'
NODE_ID_SUBINDEX = b'\x00'
SAVE_INDEX = b'\x10\x10'
SAVE_SUBINDEX = b'\x00'
SAVE_DATA = b'\x45\x56\x41\x53'

CONTROL_ENABLE = 0x01
CONTROL_FORWARD = 0x00
CONTROL_REVERSE = 0x02


class MotorController():
    transceiver = CANTransceiver()

    def __init__(self, node_id):
        self.msg_maker = CANOpenMessageFactory(debug=False)
        self.node_id = node_id

    def move_motor(self, speed: float, reverse: bool = False):
        if speed < 0 or speed > 1:
            print('invalid speed')
            return

        data = bytes()
        if self.node_id > 2:
            reverse = not reverse

        ctrl = CONTROL_ENABLE | (
            CONTROL_REVERSE if reverse else CONTROL_FORWARD)
        ctrl = ctrl.to_bytes(1)

        speed = int(2000 * speed)
        if speed < 100:
            self.send_reset()
            return
        speed = speed.to_bytes(2)[::-1]

        data += ctrl
        data += speed

        can_id, data = self.msg_maker.create_pdo_write(
            self.node_id,
            data
        )
        self.transceiver.send_CAN_message(can_id, data)

    def send_reset(self):
        # Set everything to zero
        can_id, data = self.msg_maker.create_pdo_write(
            self.node_id,
            bytes()
        )
        self.transceiver.send_CAN_message(can_id, data)

    def set_motor_mode(self, control=1, speed_reg=1, sensory=1, open_loop=1):

        value = (control) | (speed_reg << 1) | (
            sensory << 2) | (open_loop << 3)

        can_id, data = self.msg_maker.create_sdo_write(
            self.node_id,
            MOTOR_MODE_INDEX,
            MOTOR_MODE_SUBINDEX,
            value.to_bytes(1)
        )
        self.transceiver.send_CAN_message(can_id, data)

    def set_heartbeat(self, period_ms: int):
        can_id, data = self.msg_maker.create_sdo_write(
            self.node_id,
            HEARTBEAT_INDEX,
            HEARTBEAT_SUBINDEX,
            period_ms.to_bytes(2)
        )
        self.transceiver.send_CAN_message(can_id, data)

    def turn_on(self):
        can_id, data = self.msg_maker.create_start_node(self.node_id)
        self.transceiver.send_CAN_message(can_id, data)

    def set_node_id(self, new_node_id):
        can_id, data = self.msg_maker.create_sdo_write(
            self.node_id,
            NODE_ID_INDEX,
            NODE_ID_SUBINDEX,
            new_node_id.to_bytes(1)
        )
        self.transceiver.send_CAN_message(can_id, data)

    def save_settings(self):
        can_id, data = self.msg_maker.create_sdo_write(
            self.node_id,
            SAVE_INDEX,
            SAVE_SUBINDEX,
            SAVE_DATA,
            flip_endian=False
        )
        self.transceiver.send_CAN_message(can_id, data)


def main():
    conts = []
    MotorController.transceiver.read_CAN_bus()
    speed = .5
    for node_id in range(6):
        conts.append(MotorController(node_id))
        conts[-1].turn_on()
        conts[-1].set_motor_mode()
        conts[-1].move_motor(speed)

    while 1:
        MotorController.transceiver.read_CAN_bus()
        time.sleep(.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
