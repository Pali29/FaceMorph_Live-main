from capture.face_tracker import FaceTracker
from misc.linux_cam_start import linux
from morph.utils import FaceUtils
from morph.triangles import Triangulator
import cv2

def main():
    print("Starting Face Morph Live")
    tracker = FaceTracker(cam_index=1, virt_cam_device="/dev/video10", fps=30)
    tracker.run()

def test():
    tester = FaceUtils()
    img1 = tester.read_image("assets/boy_1.jpeg")
    points = tester.get_landmarks(img1)
    # landmarks = tester.draw_landmarks(img1, points)

    w, h = img1.shape[:2]
    rect = (0, 0, w, h)
    tester2 = Triangulator(rect, points)
    triangles = tester2.get_triangles(rect, points)
    triangle = tester2.draw_triangles(img1, points, triangles)

    cv2.imshow("face",triangle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # linux_module = linux()
    test()