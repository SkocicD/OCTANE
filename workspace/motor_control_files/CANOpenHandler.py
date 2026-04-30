NMT_TX = 0x000
SDO_TX = 0x600
PDO_TX = 0x200

START_NODE_SPECIFIER = b'\x01'
SDO_READ_SPECIFIER = b'\x40'


class CANOpenMessageFactory:
    def __init__(self, debug=False):
        self.debug = debug

    def create_sdo_write(self, can_node_id: int, index: bytes, subindex: bytes, value: bytes, flip_endian=True):
        can_id = SDO_TX | can_node_id
        data = bytes()
        if len(value) == 1:
            data += b'\x2f'
        elif len(value) == 2:
            data += b'\x2B'
        elif len(value) == 3:
            data += b'\x27'
        elif len(value) == 4:
            data += b'\x23'
        else:
            print("Value field too long")

        data += index[::-1]  # reversed for little endian
        data += subindex
        if flip_endian:
            data += value[::-1]  # reversed for little endian
        else:
            data += value

        data = self.pad(data)

        self.print_debug(can_id, data)
        return can_id, data

    def create_sdo_read(self, can_node_id: int, index: bytes, subindex: bytes):
        can_id = SDO_TX | can_node_id

        data = bytes()
        data += SDO_READ_SPECIFIER  # tell it to read
        data += index[::-1]  # reversed for little endian
        data += subindex

        data = self.pad(data)

        self.print_debug(can_id, data)
        return can_id, data

    def create_start_node(self, can_node_id: int):
        can_id = NMT_TX

        data = bytes()
        data += START_NODE_SPECIFIER
        data += can_node_id.to_bytes(1)

        self.print_debug(can_id, data)
        return can_id, data

    def create_pdo_write(self, can_node_id: int, data: bytes):
        can_id = PDO_TX | can_node_id
        data = self.pad(data)

        self.print_debug(can_id, data)
        return can_id, data

    @staticmethod
    def pad(data, maxlen=8):
        pad = 8 - len(data)
        data += b'\x00' * pad
        return data

    def print_debug(self, can_id, data):
        if not self.debug:
            return
        print(f'Created Message with can_id {can_id} with data ', end='')
        for d in data:
            print(f'{int(d):02x}', end=' ')
        print()
