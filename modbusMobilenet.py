import cv2
import numpy as np
import struct
import crcmod.predefined
import time
import serial

# 얼굴 감지 Mobilenet SSD 모델 로드
prototxt_path = 'deploy.prototxt.txt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 카메라 연결
cap = cv2.VideoCapture(0)

max_faces = 4  # 감지 및 추적할 최대 얼굴 수
faces_count = 0

# CRC16 초기화
crc16 = crcmod.predefined.Crc('modbus')

# 시리얼 포트 초기화 ('COM1'을 적절한 포트로 변경)
ser = serial.Serial('COM1', 9600, timeout=1)

# 이전 프레임 처리 시간 초기화
prev_frame_time = 0

# 이전 패킷 송신 시간 초기화
prev_packet_time = 0

while True:
    # 현재 시간 측정
    current_time = time.time()

    # 카메라 프레임을 프레임 단위로 캡처
    ret, frame = cap.read()

    # 프레임의 크기를 가져오기
    frame_height, frame_width = frame.shape[:2]

    # 카메라 중심 좌표를 계산
    center_x, center_y = frame_width // 2, frame_height // 2

    # 중심점 그리기
    cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), -1)

    # 프레임 중심을 (0, 0)으로 표현
    zero_x, zero_y = center_x - center_x, center_y - center_y

    # 중심점 좌표 표현
    center_text = f"Center: ({zero_x}, {zero_y})"
    cv2.putText(frame, center_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # FPS 계산
    elapsed_time = current_time - prev_frame_time
    fps = 1 / elapsed_time
    prev_frame_time = current_time

    # FPS를 화면에 표시
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 얼굴 감지를 위해 프레임을 블롭으로 변환
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    # 신경망에 입력을 설정하고 얼굴 감지를 수행
    net.setInput(blob)
    detections = net.forward()

    # 각 프레임마다 얼굴 수를 초기화
    faces_count = 0

    # 얼굴 좌표 데이터를 저장할 배열 선언
    face_data = []

    # 얼굴 좌표  데이터 수집
    for i in range(detections.shape[2]):
        if faces_count < max_faces:
            confidence = detections[0, 0, i, 2]

            if confidence > 0.9:
                faces_count += 1

                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (x, y, x2, y2) = box.astype(int)
                face_x, face_y = (x + x2) // 2, (y + y2) // 2
                value_x, value_y = face_x - center_x, center_y - face_y

                # 얼굴 좌표 데이터를 face_data 배열에 추가
                face_data.extend(struct.pack('>hh', value_x, value_y))

                # 감지된 얼굴 주위에 박싱 처리
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # 얼굴 중심점 그리기
                cv2.circle(frame, (face_x, face_y), 2, (0, 255, 0), -1)

                # 얼굴 중심점 좌표 표시
                center_coordinates_text = f"Face {faces_count}: ({value_x}, {value_y})"
                cv2.putText(frame, center_coordinates_text, (face_x + 10, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 감지된 얼굴 수 표시
    faces_text = f"faces count: {faces_count}"
    cv2.putText(frame, faces_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 패킷을 500ms마다 송신
    if current_time - prev_packet_time >= 0.5:
        # 얼굴 수를 1바이트로 변환
        count_byte = struct.pack('B', faces_count)

        # 얼굴 좌표 데이터를 맞게 수집하는지 확인
        while len(face_data) < max_faces * 4:
            # 데이터가 없으면 0으로 채움
            face_data.extend(struct.pack('>hh', 0, 0))

        # 얼굴 좌표 데이터에 대한 CRC16을 계산
        crc16.update(bytes(face_data))
        crc_value = crc16.crcValue

        # 데이터 패킷을 생성
        packet = bytearray()
        packet.append(0x02)  # STX (시작)
        packet.extend(count_byte)  # 얼굴 수 (1 바이트)
        packet.extend(face_data)  # 얼굴 좌표 데이터
        packet.extend(struct.pack('>H', crc_value))  # CRC16 (HIGH, LOW)
        packet.append(0x03)  # ETX (끝)

        # 패킷을 10진수로 변환하여 출력, 16진수 출력
        packet_decimal = ' '.join([str(byte) for byte in packet])
        packet_hex = ' '.join([hex(int(byte)) for byte in packet])
        print(f"Packet (Decimal): {packet_decimal}")
        print(f"Packet (Hex): {packet_hex}")

        # 시리얼 포트를 통해 패킷 송신
        ser.write(packet)
        ser.flush()

        # 이전 패킷 송신 시간 업데이트
        prev_packet_time = current_time

    # 얼굴 감지된 프레임 표시
    cv2.imshow("Face Detection", frame)

    # 'q'를 눌러 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 시스템 종료
cap.release()
cv2.destroyAllWindows()
