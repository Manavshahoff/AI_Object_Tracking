import sys
import os

# Add yolov5 to Python's module search path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
sys.path.append(os.path.join(os.getcwd(), 'yolov5', 'utils'))  # Add utils to the path

# YOLOv5 imports
import cv2
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Add yolov5 to Python's module search path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

# Initialize YOLO model
weights = './yolov5s.pt'  # Path to pre-trained YOLO weights
device = select_device('')  # Use GPU if available, else CPU
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Load video
# video_path = "sample_video.mp4"  # Replace with your video file
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# YOLO settings
img_size = 640
img_size = check_img_size(img_size, s=stride)  # Adjust to nearest multiple of stride

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Resize and prepare frame for YOLO
    frame_resized = cv2.resize(frame, (img_size, img_size))
    img = torch.from_numpy(frame_resized).permute(2, 0, 1).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Run inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Focus on a specific person
    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    for det in pred:
        if len(det):
            # Scale boxes to original frame size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            # Focus on the largest bounding box
            max_area = 0
            target_box = None
            for *xyxy, conf, cls in reversed(det):
                if names[int(cls)] == 'person':  # Track only "person" class
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        target_box = (x1, y1, x2, y2)

            # Highlight the target and calculate its position
            if target_box:
                x1, y1, x2, y2 = target_box
                target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, target_center, 5, (0, 0, 255), -1)

                # Calculate the offset from the frame center
                offset_x = target_center[0] - frame_center[0]
                offset_y = target_center[1] - frame_center[1]
                print(f"Offset: X={offset_x}, Y={offset_y}")

                # You can send these offsets to servo motors later to adjust camera position

    # Show video feed
    cv2.imshow('Object Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
