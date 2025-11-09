import cv2
import numpy as np
from morph.morph_core import FaceMorpher
from morph.utils import FaceUtils
import pyvirtualcam as pvc


def test_static_morph(source_path, target_path):
    # Initialize classes
    utils = FaceUtils()
    morpher = FaceMorpher()

    # Load images
    src_img = cv2.imread(source_path)
    dst_img = cv2.imread(target_path)

    if src_img is None or dst_img is None:
        print("[ERROR] Could not load one or both images.")
        return

    # Optional: resize both to same size
    dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]))

    # Get landmarks
    src_points = utils.get_landmarks(src_img)
    dst_points = utils.get_landmarks(dst_img)

    if src_points is None or dst_points is None:
        print("[ERROR] Landmark detection failed.")
        return

    # Morph at different alpha levels
    for alpha in np.linspace(0, 1, 5):
        morphed = morpher.get_morphed_face(src_img, dst_img, src_points, dst_points, alpha)
        label = f"alpha={alpha:.2f}"
        cv2.imshow(label, morphed)
        cv2.waitKey(0)
        cv2.destroyWindow(label)

    cv2.destroyAllWindows()



def test_cam_morph(cam_index=0, virt_cam_device="/dev/video10", fps=30):
    # Initialize components
    utils = FaceUtils()
    morpher = FaceMorpher()
    
    # Load target image
    target_path = "assets/faces/target.jpeg"
    target_img = cv2.imread(target_path)
    if target_img is None:
        print("[ERROR] Could not load target image.")
        return
        
    # Get target landmarks
    target_points = utils.get_landmarks(target_img)
    if target_points is None:
        print("[ERROR] Could not detect face in target image.")
        return
    
    # Initialize webcam
    video = cv2.VideoCapture(cam_index)
    ret, frame = video.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")
    height, width = frame.shape[:2]

    try:
        # Create window for keyboard events
        cv2.namedWindow('Press ESC to exit', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Press ESC to exit', 400, 100)
        
        # Initialize virtual camera
        with pvc.Camera(width=width, height=height, fps=fps, device=virt_cam_device) as cam:
            print(f"Using virtual camera: {cam.device}")
            
            alpha = 0.5  # Morphing factor
            while True:
                ret, frame = video.read()
                if not ret:
                    continue
                
                # Get landmarks for current frame
                src_points = utils.get_landmarks(frame)
                if src_points is not None and len(src_points) == len(target_points):
                    # Perform morphing
                    morphed_frame = morpher.get_morphed_face(frame, target_img, src_points, target_points, alpha)
                else:
                    morphed_frame = frame
                
                # Convert to RGB for virtual camera
                try:
                    send_frame = cv2.cvtColor(morphed_frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    send_frame = morphed_frame
                
                # Send frame to virtual camera
                cam.send(send_frame)
                
                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
                cam.sleep_until_next_frame()
                
    finally:
        video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Testing static morphing...")
    test_static_morph()
    
    print("\nTesting live camera morphing...")
    from misc.linux_cam_start import linux
    linux()  # Setup virtual camera on Linux
    test_cam_morph()
