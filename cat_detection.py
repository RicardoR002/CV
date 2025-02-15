# Import necessary libraries
import cv2
import numpy as np
import os
from datetime import datetime

# Function to create the captured_images directory if it doesn't exist
def create_capture_dir():
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")
        print("Created captured_images directory.")

# Load YOLOv4 model
# This line loads the pre-trained YOLOv4 model using OpenCV's DNN module.
net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4.cfg")

# Load COCO class names
# This file contains the names of all classes that YOLOv4 can detect.
with open("yolov4/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Layer names
# These are the names of the layers in the YOLOv4 model.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create captured_images directory if it doesn't exist
create_capture_dir()

# Camera setup
# This line initializes the camera capture object.
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    # This line captures a single frame from the camera.
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get frame dimensions
    # These variables store the height and width of the frame.
    height, width, channels = frame.shape

    # Detect objects
    # This block converts the frame into a blob and passes it through the YOLOv4 model.
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Save images when cat is detected
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Check if the detected object is a cat with high confidence
            if confidence > 0.5 and classes[class_id] == "cat":
                # Create a timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_image_{timestamp}.jpg"
                filepath = os.path.join("captured_images", filename)
                
                # Save the frame as an image
                cv2.imwrite(filepath, frame)
                print(f"Cat detected and image saved as {filename}.")

    # Display the frame
    # This line shows the current frame with any detected objects.
    cv2.imshow('Detection', frame)
    
    # Exit on key press
    # This line checks if the 'q' key is pressed and exits the loop if so.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
# This line releases the camera and closes all OpenCV windows.
cap.release()
cv2.destroyAllWindows()