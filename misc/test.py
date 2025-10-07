import numpy as np
import pyvirtualcam as pvc

frame = np.zeros((300, 200, 3), dtype=np.uint8)
frame[:10, :] = [255, 0, 0]       # Top border
frame[-10:, :] = [255, 0, 0]      # Bottom border
frame[:, :10] = [255, 0, 0]       # Left border
frame[:, -10:] = [255, 0, 0]    # Right border

with pvc.Camera(width=200, height=300, fps=20, device="/dev/video0") as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        cam.send(frame)
        cam.sleep_until_next_frame()