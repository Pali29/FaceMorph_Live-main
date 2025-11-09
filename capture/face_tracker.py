# capture/face_tracker.py

import cv2
import mediapipe as mp
import pyvirtualcam as pvc
import time
import logging

logger = logging.getLogger('face_tracker')


class FaceTracker:
    def __init__(self, cam_index=1, virt_cam_device="/dev/video10", fps=30):
        self.cam_index = cam_index
        self.virt_cam_device = virt_cam_device
        self.fps = fps

        # Initialize Mediapipe Face Mesh (kept in __init__)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.drawing_utils = mp.solutions.drawing_utils
        self.draw_specs = self.drawing_utils.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 200, 0)
        )

        # Delay opening the real webcam until run()/read() to avoid "device in use" errors
        self.video = None
        self.width = None
        self.height = None

        # Virtual camera will be opened after we know frame size
        self.cam = None

    def process_frame(self, frame):
        """Detect landmarks and draw them."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        faces = 0
        if results.multi_face_landmarks:
            faces = len(results.multi_face_landmarks)
            logger.debug(f"process_frame: detected {faces} face(s)")
            for face_landmarks in results.multi_face_landmarks:
                self.drawing_utils.draw_landmarks(
                    image=rgb,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=self.draw_specs,
                    connection_drawing_spec=self.draw_specs,
                )

        return rgb


    def open_video(self, retries=3, delay=1.0):
        """Attempt to open the webcam lazily with retries. Raises RuntimeError on failure."""
        if self.video is not None and getattr(self.video, 'isOpened', lambda: False)():
            return

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                logger.info(f"open_video: attempt {attempt} to open camera index {self.cam_index}")
                # Prefer V4L2 backend on Linux for reliability; fallback to default if unavailable
                try:
                    self.video = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
                except Exception:
                    self.video = cv2.VideoCapture(self.cam_index)

                if not self.video or not self.video.isOpened():
                    last_err = RuntimeError(f"cv2.VideoCapture could not open index {self.cam_index}")
                    # make sure to release any partial handle
                    try:
                        if self.video:
                            self.video.release()
                    except Exception:
                        pass
                    self.video = None
                    if attempt < retries:
                        logger.warning(f"open_video: could not open camera index {self.cam_index}, retrying in {delay}s")
                        time.sleep(delay)
                        continue
                    raise last_err

                # try a read to confirm camera is usable
                ret, frame = self.video.read()
                if not ret or frame is None:
                    try:
                        self.video.release()
                    except Exception:
                        pass
                    self.video = None
                    last_err = RuntimeError("Could not read a frame from webcam; it may be in use.")
                    if attempt < retries:
                        logger.warning("open_video: read failed, camera may be in use; retrying")
                        time.sleep(delay)
                        continue
                    raise last_err

                self.height, self.width = frame.shape[:2]
                logger.info(f"open_video: camera opened, resolution={self.width}x{self.height}")
                return
            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(delay)
                    continue
        # If we reach here, all attempts failed
        raise RuntimeError(
            f"Failed to open webcam at index {self.cam_index} after {retries} attempts. "
            "It may be in use by another application. Try closing other apps or use a different index."
        )


    def open_virtual_cam(self):
        """Attempt to open the virtual camera; if it fails, fall back to local display."""
        if self.cam is not None:
            return

        try:
            # pyvirtualcam expects frames in RGB by default when using send(...)
            self.cam = pvc.Camera(width=self.width, height=self.height, fps=self.fps, device=self.virt_cam_device)
            logger.info(f"Using virtual camera: {self.cam.device}")
        except Exception as e:
            # don't raise here; just warn and fall back to imshow
            logger.warning(f"could not open virtual camera '{self.virt_cam_device}': {e} - falling back to local display")
            self.cam = None

    def run(self):
        try:
            # open camera lazily (this may raise RuntimeError which we let bubble up)
            self.open_video()
        except RuntimeError as e:
            print(e)
            return

        # try virtual cam; if it fails we'll show via imshow
        self.open_virtual_cam()

        try:
            while True:
                ret, frame = self.video.read()
                if not ret or frame is None:
                    logger.warning("run: failed to read frame from camera; exiting loop")
                    break

                processed = self.process_frame(frame)

                if self.cam:
                    # pyvirtualcam expects RGB frames by default
                    try:
                        logger.debug("run: sending frame to virtual camera")
                        self.cam.send(processed)
                        self.cam.sleep_until_next_frame()
                        logger.debug("run: frame sent to virtual camera")
                    except Exception as e:
                        logger.error(f"Virtual cam error: {e} - falling back to local display.")
                        try:
                            self.cam.close()
                        except Exception:
                            pass
                        self.cam = None
                        bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                        cv2.imshow("FaceTracker", bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    cv2.imshow("FaceTracker", bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            # cleanup resources
            try:
                if self.video:
                    self.video.release()
            except Exception:
                pass
            try:
                if self.cam:
                    self.cam.close()
            except Exception:
                pass
            cv2.destroyAllWindows()
            logger.info("Stream ended.")

    def read(self):
        """Read a single frame from the webcam."""
        # open lazily if needed
        try:
            if self.video is None or not getattr(self.video, 'isOpened', lambda: False)():
                self.open_video()
        except RuntimeError:
            return None

        try:
            ret, frame = self.video.read()
            if not ret:
                return None
            return frame
        except Exception:
            return None


if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run()
