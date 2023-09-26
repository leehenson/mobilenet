import cv2
import numpy as np
import time

# Load a pre-trained MobileNet SSD model for face detection
prototxt_path = 'deploy.prototxt.txt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

max_faces = 4  # Maximum number of faces to detect and track
faces_count = 0

# Variables for calculating FPS
frame_count = 0
start_time = time.time()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    frame_count += 1

    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Calculate the center coordinates of the frame
    center_x, center_y = frame_width // 2, frame_height // 2

    # Draw a point at the center of the frame
    cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), -1)

    zero_x, zero_y = center_x - center_x, center_y - center_y

    center_text = f"Center: ({zero_x}, {zero_y})"
    cv2.putText(frame, center_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert the frame to a blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    # Set the input to the neural network and perform face detection
    net.setInput(blob)
    detections = net.forward()

    # Reset the face count for each frame
    faces_count = 0

    # Loop through the detected faces and draw bounding boxes
    for i in range(detections.shape[2]):
        if faces_count < max_faces:
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by confidence score
            if confidence > 0.9:
                faces_count += 1

                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                (x, y, x2, y2) = box.astype(int)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Calculate the center coordinates of the face
                face_x, face_y = (x + x2) // 2, (y + y2) // 2

                # Draw a point at the center of the face
                cv2.circle(frame, (face_x, face_y), 2, (0, 255, 0), -1)

                value_x, value_y = face_x - center_x, center_y - face_y

                # Display the coordinates of the face center
                center_coordinates_text = f"Face {faces_count}: ({value_x}, {value_y})"
                cv2.putText(frame, center_coordinates_text, (face_x + 10, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of faces detected
    faces_text = f"Faces Count: {faces_count}"
    cv2.putText(frame, faces_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with face detections
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
