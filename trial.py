from capture.face_tracker import FaceTracker
from morph.utils import FaceUtils
from morph.morph_core import FaceMorpher
import cv2
import os
import pyvirtualcam
import numpy as np
import threading
import time


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
    if target_points is None or len(target_points) == 0:
        raise RuntimeError("Face Not Detected")

    try:
        tracker = FaceTracker(cam_index=1, virt_cam_device="/dev/video10", fps=30)
        print("Tracker Initialised")
    except Exception as e:
        raise RuntimeError(f"Could Not initialize face tracker: {e}")

    morph_engine = FaceMorpher()
    print("Components Initialised")
    return tracker, target_img, target_points, morph_engine


def morph_live_frame(frame, target_img, target_points, morph_engine, alpha):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src_points = faceutils.get_landmarks(frame_rgb)
        if src_points is None or len(src_points) == 0:
            return frame
        if len(src_points) != len(target_points):
            return frame
        morphed = morph_engine.morph(frame_rgb, target_img, src_points, target_points, alpha)
        morphed_bgr = cv2.cvtColor(morphed, cv2.COLOR_RGB2BGR)
        return morphed_bgr
    except Exception:
        return frame


# ----------------- Threaded Morph Worker -----------------
class MorphWorker(threading.Thread):
    def __init__(self, target_img, target_points, morph_engine, alpha):
        super().__init__(daemon=True)
        self.target_img = target_img
        self.target_points = target_points
        self.morph_engine = morph_engine
        self.alpha = alpha
        self.frame_to_process = None
        self.latest_result = None
        self.lock = threading.Lock()
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process.copy()
                    self.frame_to_process = None
            if frame is not None:
                morphed = morph_live_frame(
                    frame, self.target_img, self.target_points, self.morph_engine, self.alpha
                )
                with self.lock:
                    self.latest_result = morphed
            else:
                time.sleep(0.01)

    def submit(self, frame):
        with self.lock:
            if self.frame_to_process is None:
                self.frame_to_process = frame.copy()

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def stop(self):
        self.stop_flag = True


# ----------------- Live Morph Loop -----------------
def run_live_morph(tracker, target_img, target_points, morph_engine, virt_cam_device="/dev/video10", fps=30):
    frame = tracker.read()
    if frame is None:
        raise RuntimeError("Could not read initial frame")
    height, width = frame.shape[:2]

    alpha = 0.5
    worker = MorphWorker(target_img, target_points, morph_engine, alpha)
    worker.start()

    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, device=virt_cam_device) as cam:
            print("Virtual cam started")

            while True:
                frame = tracker.read()
                if frame is None:
                    continue

                # Submit frame to worker
                worker.submit(frame)

                # Get latest processed result
                result = worker.get_latest()
                if result is None:
                    result = frame

                cam.send(result)
                cam.sleep_until_next_frame()

                cv2.imshow("FaceMorph Live", result)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

    except Exception as e:
        print(f"Virtual cam failed: {e}")

    finally:
        worker.stop()
        cv2.destroyAllWindows()


# ----------------- Entry Point -----------------
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
