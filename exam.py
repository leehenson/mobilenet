import struct
import crcmod.predefined
import time
import serial  # 필요한 경우 시리얼 통신 라이브러리를 설치하세요.

# 시리얼 포트 설정 (COM1 또는 /dev/ttyS0와 같이 적절한 포트로 변경해야 함)
ser = serial.Serial('COM1', 9600, timeout=1)

# CRC16 계산을 위한 CRC 객체 생성
crc16 = crcmod.predefined.Crc('modbus')

while True:
    # 변수 설정
    count = 4
    x1 = -231
    y1 = 132
    x2 = 24
    y2 = -29
    x3 = 234
    y3 = 4
    x4 = 25
    y4 = -234
    
    # 패킷 구성
    packet = bytearray()
    packet.append(0x02)  # STX
    packet.append(count)  # 변수 count
    for value in [x1, y1, x2, y2, x3, y3, x4, y4]:
        packet.extend(struct.pack('>h', value))  # signed 2바이트 값 추가
    # CRC16 계산
    crc16.update(packet)
    crc_value = crc16.crcValue
    packet.extend(struct.pack('>H', crc_value))  # CRC16 High, Low
    packet.append(0x03)  # ETX
    
    # 시리얼 포트를 통해 패킷 전송
    ser.write(packet)
    ser.flush()
    print(packet)
    
    # 500ms 간격으로 전송
    time.sleep(0.5)
