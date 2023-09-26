"""
def calculate_crc16(data):
    crc = 0xFFFF
    polynomial = 0xA001

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1

    return crc
"""

def calculate_crc16(data):
    crc = 0xFFFF
    polynomial = 0xA001

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1

    # Swap the bytes before returning
    crc = ((crc & 0xFF) << 8) | ((crc >> 8) & 0xFF)

    return crc


msg = bytes.fromhex("02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00")
#crc = modbusCrc(msg)
crc = calculate_crc16(msg)
print("0x%04X"%(crc))            

ba = crc.to_bytes(2, byteorder='little')
print("%02X %02X"%(ba[0], ba[1]))