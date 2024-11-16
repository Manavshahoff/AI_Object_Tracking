import cv2

# Replace with your video file path
video_path = "sample_video.mp4"  # Update if necessary
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video or error")
        break

    # Display the frame in a window
    cv2.imshow('Video Feed', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Adjust waitKey for frame rate
        break

cap.release()
cv2.destroyAllWindows()
