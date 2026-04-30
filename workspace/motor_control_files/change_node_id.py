from MotorController import MotorController
cont = MotorController(0x01)
cont.set_node_id(0x04)
cont.set_heartbeat(1000)
cont.save_settings()
