NMT_TX = 0x000
SDO_TX = 0x600
PDO_TX = 0x200


class CANOpenMessageFactory:

    def create_sdo_write(self, node_id: int, index: bytes, subindex: bytes, value: bytes) -> tuple:
        can_id = SDO_TX | node_id
        specifier = {1: b'\x2f', 2: b'\x2B', 3: b'\x27', 4: b'\x23'}.get(len(value))
        if specifier is None:
            raise ValueError(f'SDO value length {len(value)} not supported')
        data = specifier + index[::-1] + subindex + value[::-1]
        return can_id, self._pad(data)

    def create_sdo_read(self, node_id: int, index: bytes, subindex: bytes) -> tuple:
        can_id = SDO_TX | node_id
        data = b'\x40' + index[::-1] + subindex
        return can_id, self._pad(data)

    def create_nmt_start(self, node_id: int) -> tuple:
        return NMT_TX, self._pad(b'\x01' + node_id.to_bytes(1, 'little'))

    def create_pdo_write(self, node_id: int, data: bytes) -> tuple:
        return PDO_TX | node_id, self._pad(data)

    @staticmethod
    def _pad(data: bytes, length: int = 8) -> bytes:
        return data + b'\x00' * (length - len(data))
