import numpy as np
import cv2
from morph.triangles import Triangulator
from morph.utils import FaceUtils


class FaceMorpher:
    def __init__(self):
        self.utils = FaceUtils
        self.triangulator = Triangulator

    
    def warp_triangle(self, img, t_in, t_out):
        t_in = np.array(t_in, dtype=np.float32)
        t_out = np.array(t_out, dtype=np.float32)

        r_in = cv2.boundingRect(t_in)
        r_out = cv2.boundingRect(t_out)

        t_in_offset = t_in - [r_in[0], r_in[1]]
        t_out_offset = t_out - [r_out[0], r_out[1]]

        img_crop = img[r_in[1]:r_in[1]+r_in[3], r_in[0]:r_in[0]+r_in[2]]
        mask = np.zeros((r_out[3], r_out[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t_out_offset), 255)

        t_in_offset = np.array(t_in_offset, dtype=np.float32)
        t_out_offset = np.array(t_out_offset, dtype=np.float32)

        # skip degenerate triangles
        if t_in.shape[0] != 3 or t_out.shape[0] != 3:
            return np.zeros((0,0), dtype=np.uint8), np.zeros((0,0), dtype=np.uint8), (0,0,0,0)


        warp_mat = cv2.getAffineTransform(t_in_offset, t_out_offset)
        warped_patch = cv2.warpAffine(img_crop, warp_mat, (r_out[2], r_out[3]), flags= cv2.INTER_LINEAR, borderMode= cv2.BORDER_REFLECT_101)
        
        return warped_patch, mask, r_out
    

    def morph_triangle(self, img1, img2, img_morph, t1, t2, t, alpha):
        warp1, mask1, r1 = self.warp_triangle(img1, t1, t)
        warp2, mask2, r2 = self.warp_triangle(img2, t2, t)

        blended_patch = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)
        blended_patch = blended_patch.astype(img_morph.dtype)
        
        x,y,w,h = r1
        x = max(x, 0)
        y = max(y, 0)
        roi = img_morph[y:y+h, x:x+w]

        mask_out = cv2.merge([mask1, mask1, mask1]) if len(img_morph.shape) == 3 else mask1

        img_morph[y:y+h, x:x+w] = roi * (1-mask_out/255.0) + blended_patch * (mask_out/255.0)


    def get_morphed_face(self, src_img, dst_img, src_points, dst_points, alpha):
        morphed_img = src_img.astype(np.uint8)

        alpha = np.clip(alpha, 0.0, 1.0)

        interpolated_points = []
        for i in range(len(src_points)):
            x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
            y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
            interpolated_points.append((x, y))

        interpolated_points = np.array(interpolated_points, dtype=np.float32)

        h, w = morphed_img.shape[:2]
        morphed_img_rect = (0, 0, h, w)

        triangulater = self.triangulator(morphed_img_rect, interpolated_points)
        triangles = triangulater.get_triangles(morphed_img_rect, interpolated_points)

        for tri_indices in triangles:
            x, y, z = tri_indices

            t1 = [src_points[x], src_points[y], src_points[z]]
            t2 = [dst_points[x], dst_points[y], dst_points[z]]
            t = [interpolated_points[x], interpolated_points[y], interpolated_points[z]]

            self.morph_triangle(src_img, dst_img, morphed_img, t1, t2, t, alpha)

        return morphed_img
    

    def blend_faces(self, face1, face2, alpha):
        blended_face = cv2.addWeighted(face1, 1-alpha, face2, alpha, 0)
        blended_face = blended_face.astype(np.uint8)

        return blended_face