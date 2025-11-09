import cv2

# Initialize webcam (0 = default camera)
cam = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cam.isOpened():
    print("Error: Cannot access webcam")
    exit()

# Read a single frame
ret, frame = cam.read()

if ret:
    # Save the captured image
    cv2.imwrite("assets/Test1.jpg", frame)
    print("Image captured and saved as Test1.jpg")
else:
    print("Error: Failed to capture image")

# Release the webcam
cam.release()

# Close all OpenCV windows (optional)
cv2.destroyAllWindows()
