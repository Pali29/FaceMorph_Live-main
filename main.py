from capture.face_tracker import FaceTracker
from morph.utils import FaceUtils
from morph.morph_core import FaceMorpher
import cv2
import os
import pyvirtualcam
import threading
import logging

# basic logger for diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('vibe')


faceutils = FaceUtils()

def init_components():
    target_path = "assets/Test1.jpg"
    if not os.path.exists(target_path):
        raise FileNotFoundError("no target detected")
    
    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError("Error reading Image")
    # keep as BGR here; FaceUtils.get_landmarks expects BGR and will convert internally

    target_points = faceutils.get_landmarks(target_img)
    if target_points is None or len(target_points)==0:
        raise RuntimeError("Face Not Detected")

    try:
        tracker = FaceTracker()
        logger.info("Tracker initialised")
    except Exception as e:
        # If tracker cannot initialize we'll raise so the caller handles it
        logger.error(f"Tracker init error: {e}")
        tracker = None

    if tracker is None:
        raise RuntimeError("Could not initialize FaceTracker")
    
    morph_engine = FaceMorpher()

    logger.info("Components Initialised")
    return tracker, target_img, target_points, morph_engine


def morph_live_frame(frame, target_img, target_points, morph_engine, alpha):
    try:
        # leave frame in BGR; FaceUtils.get_landmarks expects BGR and will convert internally
        src_points = faceutils.get_landmarks(frame)
        if src_points is None or len(src_points)==0:
            return frame
        
        if len(src_points) != len(target_points):
            # Landmark count mismatch — cannot reliably morph. Log counts for debugging.
            logger.warning(f"Landmark count mismatch: src={len(src_points)} target={len(target_points)}")
            return frame
        # FaceMorpher provides get_morphed_face (not .morph)
        morphed = morph_engine.get_morphed_face(frame, target_img, src_points, target_points, alpha)

        # get_morphed_face returns an image with the same color ordering as inputs (BGR here)
        return morphed
    
    except Exception as e:
        print(e)
        print("Morphing not done")
        return frame
    


def run_live_morph(tracker, target_img, target_points, morph_engine, virt_cam_devices="/dev/video10", fps=30):
    frame = tracker.read()
    if frame is None:
        raise RuntimeError("Could not read Initial frame")
    
    height, width = frame.shape[:2]

    try:
        # Create a small window to capture keyboard events
        cv2.namedWindow('Press ESC to exit', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Press ESC to exit', 400, 100)
        
        with pyvirtualcam.Camera(width = width, height = height, fps = fps, device = virt_cam_devices) as cam:
            alpha = 0.5
            while True:
                frame = tracker.read()
                if frame is None:
                    continue

                # worker = threading.Thread(target=task, args=(cam, frame, target_img, target_points, morph_engine, alpha), daemon=True)
                # worker.start()

                morphed_frame = morph_live_frame(frame, target_img, target_points, morph_engine, alpha)

                # pyvirtualcam expects RGB frames by default; convert from BGR
                try:
                    send_frame = cv2.cvtColor(morphed_frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    send_frame = morphed_frame

                # log before sending the frame
                try:
                    logger.debug(f"Sending frame to virtual camera, shape={getattr(send_frame, 'shape', None)}")
                    cam.send(send_frame)
                    logger.debug("Frame sent to virtual camera")
                except Exception as e:
                    logger.error(f"Failed to send frame to virtual camera: {e}")

                # Check for ESC key (must have a window open for this to work)
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("ESC pressed, exiting...")
                    break
                    
                cam.sleep_until_next_frame()

    except Exception as e:
        logger.error(f"Virtual cam error: {e}")

    finally:
        cv2.destroyAllWindows()



def task(cam, frame, target_img, target_points, morph_engine, alpha):
    morphed_frame = morph_live_frame(frame, target_img, target_points, morph_engine, alpha)

    # pyvirtualcam expects RGB frames by default; convert from BGR
    try:
        send_frame = cv2.cvtColor(morphed_frame, cv2.COLOR_BGR2RGB)
    except Exception:
        send_frame = morphed_frame

    # log before sending the frame
    try:
        logger.debug(f"Sending frame to virtual camera, shape={getattr(send_frame, 'shape', None)}")
        cam.send(send_frame)
        logger.debug("Frame sent to virtual camera")
    except Exception as e:
        logger.error(f"Failed to send frame to virtual camera: {e}")

    # cam.sleep_until_next_frame()



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
    from misc.linux_cam_start import linux
    linux()
    main()