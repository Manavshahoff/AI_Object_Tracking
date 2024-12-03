import sys
import os
import cv2
import torch
import serial  # type: ignore # Import for serial communication
import time  # To manage delays

# Add yolov5 to Python's module search path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
sys.path.append(os.path.join(os.getcwd(), 'yolov5', 'utils'))  # Add utils to the path

# YOLOv5 imports
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Initialize YOLO model
weights = './yolov5s.pt'  # Path to pre-trained YOLO weights
device = select_device('')  # Use GPU if available, else CPU
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Set up serial communication with Arduino
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)  # Replace 'COM4' with your Arduino's port
time.sleep(2)  # Wait for the Arduino to initialize

# Send initial command to set servos to neutral position
try:
    arduino.write("0,0\n".encode())  # Move servos to the center position
    print("Motors initialized to the center position.")
except Exception as e:
    print(f"Error initializing motors: {e}")

# Load video
cap = cv2.VideoCapture(1)  # Use camera input
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# YOLO settings
img_size = 640
img_size = check_img_size(img_size, s=stride)  # Adjust to nearest multiple of stride
no_human_count = 0  # Counter to track frames with no human detected
NO_HUMAN_RESET_THRESHOLD = 30  # Reset to neutral after 30 frames with no human detected
MOVEMENT_THRESHOLD = 10  # Minimum offset threshold to avoid small, unnecessary movements
SCALE_FACTOR = 15  # Scale offsets to control servo speed

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Flip the frame horizontally to correct for mirroring
    frame = cv2.flip(frame, 1)

    # Resize and prepare frame for YOLO
    frame_resized = cv2.resize(frame, (img_size, img_size))
    img = torch.from_numpy(frame_resized).permute(2, 0, 1).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Run inference and filter for "person" class
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[0])  # Class 0 = "person"

    # Focus on detected persons
    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    largest_area = 0
    target_center = None

    for det in pred:
        if len(det):
            # Scale boxes to original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Find the largest bounding box (track closest person)
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    if target_center:
        # Human detected, reset no-human counter
        no_human_count = 0

        # Calculate the offset from the frame center
        offset_x = target_center[0] - frame_center[0]
        offset_y = target_center[1] - frame_center[1]

        # Debug: Print offset values
        print(f"Frame Center: {frame_center}, Target Center: {target_center}")
        print(f"Offset: X={offset_x}, Y={offset_y}")

        # Apply movement threshold and keep the human at the center of the frame
        if abs(offset_x) > MOVEMENT_THRESHOLD or abs(offset_y) > MOVEMENT_THRESHOLD:
            scaled_offset_x = offset_x // SCALE_FACTOR  # Scale X for smooth movement
            scaled_offset_y = offset_y // SCALE_FACTOR  # Scale Y for smooth movement

            # Reverse offsets to align servo movement with target direction
            try:
                arduino.write(f"{-scaled_offset_x},{-scaled_offset_y}\n".encode())
                print(f"Sent to Arduino: X={-scaled_offset_x}, Y={-scaled_offset_y}")
            except Exception as e:
                print(f"Error sending data to Arduino: {e}")

        # Draw bounding box and center mark
        cv2.rectangle(frame, (target_center[0] - 10, target_center[1] - 10),
                      (target_center[0] + 10, target_center[1] + 10), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # No human detected, increment no-human counter
        no_human_count += 1
        if no_human_count > NO_HUMAN_RESET_THRESHOLD:
            # Reset servos to neutral position
            try:
                arduino.write("0,0\n".encode())
                print("No human detected, resetting to center.")
            except Exception as e:
                print(f"Error resetting motors: {e}")

            cv2.putText(frame, "No human detected - Resetting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show video feed
    cv2.imshow('Human Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
