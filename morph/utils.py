import cv2
import numpy as np
import mediapipe as mp

class FaceUtils():
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.facemesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def read_image(self,path,size=(600,600)):
        img=cv2.imread(path)

        if img is None:
            raise ValueError(f"could not load image: {path}")
        img = cv2.resize(img, size)
        return img
    
    def get_landmarks(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.facemesh.process(rgb)

        h,w,_ = img.shape
        points = []
        if results.multi_face_landmarks:
            for landmark in results.multi_face_landmarks[0].landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
        
        return np.array(points, np.int32)
        
    def draw_landmarks(self, img, points, color=(0,200,0)):
        for (x, y) in points:
            cv2.circle(img, (x, y), 1, color, -1)
        return img