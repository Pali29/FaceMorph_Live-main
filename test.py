import cv2
import numpy as np
from morph.morph_core import FaceMorpher
from morph.utils import FaceUtils


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


if __name__ == "__main__":
    test_static_morph()
