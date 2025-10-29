import cv2

# Open the default webcam (0 = first camera)
cap = cv2.VideoCapture(1)

# Read one frame to make sure it's initialized
ret, frame = cap.read()

if ret:
    # Get frame dimensions
    height, width, channels = frame.shape
    print(f"Webcam frame size: {width} x {height} pixels")
else:
    print("Failed to capture frame.")

# Release camera
cap.release()
