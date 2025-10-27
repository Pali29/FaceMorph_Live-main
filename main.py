from capture.face_tracker import FaceTracker
from morph.utils import FaceUtils
from morph.morph_core import FaceMorpher
import cv2
import os
import pyvirtualcam
import numpy as np


faceutils = FaceUtils()

def init_components():
    target_path = "assets/faces/source.jpeg"
    if not os.path.exists(target_path):
        raise FileNotFoundError("no target detected")
    
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError("Error reading Image")
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    target_points = faceutils.get_landmarks(target_img)
    if target_points is None or len(target_points)==0:
        raise RuntimeError("Face Not Detected")

    try:
        tracker = FaceTracker()
        print("Tracker Initialised")
    except Exception as e:
        # raise RuntimeError("Could Not initialize face tracker")
        print(e)
    
    morph_engine = FaceMorpher()

    print("Components Initialised")
    return tracker, target_img, target_points, morph_engine


def morph_live_frame(frame, target_img, target_points, morph_engine, alpha):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        src_points = faceutils.get_landmarks(frame_rgb)
        if src_points is None or len(src_points)==0:
            return frame
        
        if len(src_points) != len(target_points):
            return frame
        
        morphed = morph_engine.morph(frame_rgb, target_img, src_points, target_points, alpha)

        morphed_bgr = cv2.cvtColor(morphed, cv2.COLOR_RGB2BGR)
        return morphed_bgr
    
    except Exception as e:
        return frame
    


def run_live_morph(tracker, target_img, target_points, morph_engine, virt_cam_devices="/dev/video10", fps=30):
    frame = tracker.read()
    if frame is None:
        raise RuntimeError("Could not read Initial frame")
    
    height, width = frame.shape[:2]

    try:
        with pyvirtualcam.Camera(width = width, height = height, fps = fps, device = virt_cam_devices) as cam:
            alpha = 0.5
            while True:
                frame = tracker.read()
                if frame is None:
                    continue

                morphed_frame = morph_live_frame(frame, target_img, target_points, morph_engine, alpha)
                
                cam.send(morphed_frame)
                cam.sleep_until_next_frame()

                if cv2.waitkey(1) & 0XFF == 27:
                    break

    except Exception as e:
        # print("Virtual cam failed")
        print(e)

    finally:
        cv2.destroyAllWindows


def main():
    try:
        tracker, target_img, target_points, morph_engine = init_components()

        run_live_morph(tracker, target_img, target_points, morph_engine)

    except Exception as e:
        print(e)
    
    # finally:
    #     if "tracker" in locals() and tracker is not None:
    #         tracker.release()
    

if __name__ == "__main__":
    main()