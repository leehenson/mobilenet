import serial

# CNCB1에 연결 (적절한 포트로 변경)
ser = serial.Serial('COM4', 9600, timeout=10)

while True:
    if ser.inWaiting() > 20:  # 최소 패킷 크기가 21 바이트임
        packet = ser.read(21)  # 한 번에 하나의 패킷만 읽음
        print(f"Received packet: {packet}")
