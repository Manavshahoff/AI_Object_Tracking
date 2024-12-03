import cv2
cap = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 or 2 for other devices

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Test Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
