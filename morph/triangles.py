import cv2
import mediapipe as mp
import numpy as np

class Triangulator():
    def __init__(self, rect, points):
        self.rect = rect
        self.points = points


    def rect_contains(rect, point):
        x, y, w, h = rect
        px, py = point
        return x<=px<=x+w and y<=py<=y+h
    
    
    def get_triangles(self, rect, points):
        rect_area = cv2.Subdiv2D(rect)
        for point in points:
            rect_area.insert(point)
        
        triangle_list = rect_area.getTriangleList()
        triangle_list = np.array(triangle_list, dtype=np.float32)
        
        index_triangles = []

        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

            if not (self.rect_contains(rect, (t[0], t[1])) and self.rect_contains(rect, (t[2], t[3])) and self.rect_contains(rect, (t[4], t[5]))):
                continue

            indices = []

            for pt in pts:
                min_dist = float('inf')
                index = -1

                for i, landmark in enumerate(points):
                    dist = np.linalg.norm(np.array(pt) - np.arrya(landmark))
                    if dist < min_dist:
                        min_dist = dist
                        index = i

                if min_dist < 1.0:
                    indices.append(index)
                
            if len(indices) == 3:
                if len(set(indices)) == 3:
                    index_triangles.append(tuple(indices))

        return index_triangles
    
    
    def draw_triangles(self, img, points, triangles):
        for tri in triangles:
            pt1 = points[tri[0]]
            pt2 = points[tri[1]]
            pt3 = points[tri[2]]

            cv2.line(img, pt2, pt1, (0,200,0), 1)
            cv2.line(img, pt3, pt2, (0,200,0), 1)
            cv2.line(img, pt1, pt3, (0,200,0), 1)