import numpy as np
import cv2
from morph.triangles import Triangulator
from morph.utils import FaceUtils


class FaceMorpher:
    def __init__(self):
        self.utils = FaceUtils()
        self.triangulator = Triangulator()

    
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

        warp_mat = cv2.getAffineTransform(t_in_offset, t_out_offset)
        warped_patch = cv2.warpAffine(img_crop, warp_mat, (r_out[2], r_out[3]), flags= cv2.INTER_LINEAR, borderMode= cv2.BORDER_REFLECT_101)
        
        return warped_patch, mask, r_out
    

    def morph_triangle(self, img1, img2, img_morph, t1, t2, t, alpha):
        warp1, mask1, r1 = self.warp_triangle(img1, t1, t)
        warp2, mask2, r2 = self.warp_triangle(img2, t2, t)

        blended_patch = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)
        blended_patch = blended_patch.astype(img_morph.dtype)
        
        x,y,w,h = r1
        roi = img_morph[y:y+h, x:x+w]

        mask_out = cv2.merge([mask1, mask1, mask1]) if len(img_morph.shape) == 3 else mask1

        img_morph[y:y+h, x:x+w] = roi * (1-mask_out/255.0) + blended_patch * (mask_out/255.0)