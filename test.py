import cv2
import numpy as np
from morph.morph_core import FaceMorpher
from morph.utils import FaceUtils
import mediapipe as mp
import pyvirtualcam as pvc


def test_static_morph():
    # Initialize classes
    utils = FaceUtils()
    morpher = FaceMorpher()

    # Load images (make sure both are same size or will be resized)
    src_path = "assets/source.png"
    dst_path = "assets/target.png"

    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

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



def test_cam_morph(self, cam_index=1, virt_cam_device="/dev/video10", fps=30):
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

    utils = FaceUtils()
    morpher = FaceMorpher()

    dst_img = "assets/faces/source.png"
    
    while True:
        ret, frame = self.video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)



if __name__ == "__main__":
    test_static_morph()
