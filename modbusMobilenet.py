import cv2
import numpy as np
import time
import serial

# modbus CRC16 계산 함수
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

# 시리얼 포트 연결
ser = serial.Serial('COM1', 9600)  # 'COM1'을 연결할 시리얼 포트로 교체하면서 사용해야 함.

# 얼굴 감지 Mobilenet SSD 모델 로드
prototxt_path = 'deploy.prototxt.txt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 카메라 연결
cap = cv2.VideoCapture(0)

max_faces = 4  # 감지할 최대 얼굴 수
faces_count = 0

# FPS를 계산하기 위한 변수 선언
frame_count = 0
start_time = time.time()

# 패킷 전송 타이밍을 위한 변수 선언
packet_interval = 0.5  # 500ms 간격
last_packet_time = time.time()

while True:
    # 카메라에서 프레임 단위로 캡처
    ret, frame = cap.read()

    frame_count += 1

    # 폭, 높이 값 가져오기
    frame_height, frame_width = frame.shape[:2]

    # 중심 좌표값
    center_x, center_y = frame_width // 2, frame_height // 2

    # 중심에 점 생성
    cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), -1)

    # 중심 좌표값을 (0, 0)으로 설정
    zero_x, zero_y = center_x - center_x, center_y - center_y

    # 중심 좌표값 화면에 표시
    center_text = f"Center: ({zero_x}, {zero_y})"
    cv2.putText(frame, center_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 프레임을 blob으로 변환
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    # 얼굴 인식 실행
    net.setInput(blob)
    detections = net.forward()

    # 인식 얼굴 수 초기화
    faces_count = 0

    # 얼굴 좌표 데이터 배열 초기화
    value_x_list = []
    value_y_list = []

    # 감지된 얼굴 객체에 박싱처리
    for i in range(detections.shape[2]):
        if faces_count < max_faces:
            confidence = detections[0, 0, i, 2]

            # 90% 이상의 정확도를 가진 객체로만 필터링
            if confidence > 0.9:
                faces_count += 1

                # 박스의 좌표값 가져오기
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (x, y, x2, y2) = box.astype(int)

                # 감지된 얼굴 객체에 박스 표시
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # 감지된 얼굴 객체 중심 좌표 계산
                face_x, face_y = (x + x2) // 2, (y + y2) // 2

                # 감지된 얼굴 객체 중심점 표시
                cv2.circle(frame, (face_x, face_y), 2, (0, 255, 0), -1)

                # 감지된 얼굴 객체 중심점 좌표값 계산
                value_x, value_y = face_x - center_x, center_y - face_y

                # 얼굴 좌표 데이터 배열에 감지된 얼굴 객체 좌표 데이터 추가
                value_x_list.append(value_x)
                value_y_list.append(value_y)

                # 감지된 얼굴 객체에 얼굴 중심점 좌표 데이터 표시
                center_coordinates_text = f"Face {faces_count}: ({value_x}, {value_y})"
                cv2.putText(frame, center_coordinates_text, (face_x + 10, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    # 감지된 얼굴 수 표시
    faces_text = f"Faces Count: {faces_count}"
    cv2.putText(frame, faces_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS 값 표시
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 얼굴 감지된 프레임 띄움
    cv2.imshow("Face Detection", frame)

    # 500ms마다 패킷 송신
    current_time = time.time()
    if current_time - last_packet_time >= packet_interval:
        # 전송할 패킷 초기화
        packet = [0x02, faces_count]

        # 최대 4개의 얼굴 좌표 데이터를 얼굴 좌표 데이터 배열에 추가
        for i in range(4):
            if i < len(value_x_list):
                packet.extend([(value_x_list[i] >> 8) & 0xFF, value_x_list[i] & 0xFF])
            else:
                packet.extend([0, 0])  # 감지가 안된 데이터는 0으로 채움

            if i < len(value_y_list):
                packet.extend([(value_y_list[i] >> 8) & 0xFF, value_y_list[i] & 0xFF])
            else:
                packet.extend([0, 0])  # 감지가 안된 데이터는 0으로 채움

        crc = calculate_crc16(packet)
        packet.extend([crc >> 8, crc & 0xFF, 0x03])

        # 송신한 데이터를 16진수로 변환하여 콘솔로 출력
        print("Transmitted Packet (Bytes):", ' '.join(['{:02X}'.format(byte) for byte in packet]))

        # 패킷 전송
        ser.write(bytearray(packet))

        # 패킷 시간 업데이트
        last_packet_time = current_time

    # q키를 눌러 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 프로그램 종료
cap.release()
ser.close()
cv2.destroyAllWindows()
