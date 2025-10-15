# capture/face_tracker.py

import cv2
import mediapipe as mp
import pyvirtualcam as pvc


class FaceTracker:
    def __init__(self, cam_index=1, virt_cam_device="/dev/video10", fps=30):
        self.cam_index = cam_index
        self.virt_cam_device = virt_cam_device
        self.fps = fps

        # Initialize Mediapipe Face Mesh
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

        # Initialize Webcam
        self.video = cv2.VideoCapture(self.cam_index)
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Could not read from webcam.")
        self.height, self.width, _ = frame.shape

        # Initialize Virtual Camera
        self.cam = pvc.Camera(
            width=self.width,
            height=self.height,
            fps=self.fps,
            device=self.virt_cam_device,
        )
        print(f"Using virtual camera: {self.cam.device}")

    def process_frame(self, frame):
        """Detect landmarks and draw them."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.drawing_utils.draw_landmarks(
                    image=rgb,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=self.draw_specs,
                    connection_drawing_spec=self.draw_specs,
                )
        return rgb

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            processed = self.process_frame(frame)
            self.cam.send(processed)
            self.cam.sleep_until_next_frame()

        self.video.release()
        print("Stream ended.")


if __name__ == "__main__":
    tracker = FaceTracker()
    tracker.run()
