import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
model = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel'
)

# Define the classes that the model can detect
classes = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

max_people = 4
people_count = 0

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # Add a reference point at the center of the screen
    c_center_x, c_center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.circle(frame, (c_center_x, c_center_y), 2, (255, 0, 0), -1)

    cs_center_x, cs_center_y = c_center_x - c_center_x, c_center_y - c_center_y

    # Display the coordinates of the reference point
    coordinates_text = f"Center: ({cs_center_x}, {cs_center_y})"
    cv2.putText(frame, coordinates_text, (c_center_x + 10, c_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    model.setInput(blob)
    detections = model.forward()

    # Reset the people count for each frame
    people_count = 0

    # Loop through the detections and draw boxes around detected people
    for i in range(detections.shape[2]):
        class_id = int(detections[0, 0, i, 1])
        if class_id == 15:  # Class ID for "person"
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8 and people_count < max_people:
                people_count += 1
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate the center coordinates of the bounding box
                center_x, center_y = (startX + endX) // 2, (startY + endY) // 2
                
                # Draw a point at the center
                cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)
                
                o_center_x, o_center_y = center_x - c_center_x, c_center_y - center_y

                # Display the coordinates of the center
                center_coordinates_text = f"Object {people_count} : ({o_center_x}, {o_center_y})"
                cv2.putText(frame, center_coordinates_text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence * 100:.2f}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of people detected
    people_text = f"People Count: {people_count}"
    cv2.putText(frame, people_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame with person detections
    cv2.imshow("Person Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
