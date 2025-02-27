# presentation_controller.py
import cv2
import numpy as np
import pyautogui as gui
import time

# Initialize GUI controls with zero delay
gui.PAUSE = 0

# Face detection model paths
model_path = './model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_path = './model/deploy.prototxt'

def detect_faces(net, frame):
    """Detect faces using pre-trained CNN model
    Args:
        net: Loaded DNN model
        frame: Webcam video frame
    Returns:
        List of detected faces with coordinates and confidence
    """
    detected_faces = []
    (h, w) = frame.shape[:2]
    
    # Preprocess frame for model input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, 
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Extract face coordinates with confidence > 50%
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({
                'start': (startX, startY),
                'end': (endX, endY),
                'confidence': confidence
            })
    return detected_faces

def map_to_presentation_controls(detected_faces, frame, bbox):
    """Convert face position to presentation commands
    Args:
        detected_faces: List of detected faces
        frame: Webcam frame for visual feedback
        bbox: Control zone boundaries
    """
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        face_center_x = (x1 + x2) // 2

        # Horizontal slide control
        if x1 < bbox[0]:
            gui.press('left')  # Previous slide
            cv2.putText(frame, 'PREV SLIDE', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        elif x2 > bbox[1]:
            gui.press('right')  # Next slide
            cv2.putText(frame, 'NEXT SLIDE', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Vertical zoom control
        if y1 < bbox[3]:
            gui.hotkey('ctrl', '+')  # Zoom in
            cv2.putText(frame, 'ZOOM IN', (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        elif y2 > bbox[2]:
            gui.hotkey('ctrl', '-')  # Zoom out
            cv2.putText(frame, 'ZOOM OUT', (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

def initialize_system():
    """Set up presentation control environment"""
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Define control zones (left, right, bottom, top)
    bbox = [
        frame_width//2 - 200,  # Left boundary
        frame_width//2 + 200,  # Right boundary
        frame_height//2 + 150, # Bottom boundary
        frame_height//2 - 150  # Top boundary
    ]
    
    return net, cap, bbox

def main_control_loop():
    """Main presentation control loop"""
    net, cap, bbox = initialize_system()
    last_action = time.time()
    action_delay = 1  # Seconds between actions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        detected_faces = detect_faces(net, frame)
        
        # Visual feedback layers
        cv2.rectangle(frame, (bbox[0], bbox[3]), 
                     (bbox[1], bbox[2]), (0,0,255), 2)
        cv2.putText(frame, "Control Zone", (bbox[0]+10, bbox[3]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # Throttle control actions
        if time.time() - last_action > action_delay:
            map_to_presentation_controls(detected_faces, frame, bbox)
            last_action = time.time()
        
        cv2.imshow('Presentation Controller', frame)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start presentation in fullscreen (adjust for your software)
    gui.hotkey('f5')
    time.sleep(1)  # Allow presentation to launch
    main_control_loop()