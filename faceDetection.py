import cv2
import numpy as np

# Load a pre-trained face detection model (e.g., Haar Cascade or a deep learning-based face detector)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

max_faces = 4  # Maximum number of faces to detect and track
faces_count = 0

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # Get the dimensions of the frame
    frame_height, frame_width = frame.shape[:2]

    # Calculate the center coordinates of the frame
    center_x, center_y = frame_width // 2, frame_height // 2

    # Draw a point at the center of the frame
    cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), -1)

    zero_x, zero_y = center_x - center_x, center_y - center_y

    center_text = f"Center: ({zero_x}, {zero_y})"
    cv2.putText(frame, center_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Reset the face count for each frame
    faces_count = 0

    # Loop through the detected faces and draw bounding boxes
    for (x, y, w, h) in faces:
        if faces_count < max_faces:
            faces_count += 1

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center coordinates of the face
            face_x, face_y = (x + x + w) // 2, (y + y + h) // 2

            # Draw a point at the center of the face
            cv2.circle(frame, (face_x, face_y), 2, (0, 255, 0), -1)

            # Display the coordinates of the face center
            center_coordinates_text = f"Face {faces_count}: ({face_x - center_x}, {center_y - face_y})"
            cv2.putText(frame, center_coordinates_text, (face_x + 10, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of faces detected
    faces_text = f"Faces Count: {faces_count}"
    cv2.putText(frame, faces_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with face detections
    cv2.imshow("Face Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
