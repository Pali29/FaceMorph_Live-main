import numpy as np
import cv2
from morph.triangles import Triangulator
from morph.utils import FaceUtils
import logging

logger = logging.getLogger('morph_core')


class FaceMorpher:
    def __init__(self):
        self.utils = FaceUtils
        self.triangulator = Triangulator

    
    def warp_triangle(self, img, t_in, t_out):
        t_in = np.array(t_in, dtype=np.float32)
        t_out = np.array(t_out, dtype=np.float32)

        r_in = cv2.boundingRect(t_in)
        r_out = cv2.boundingRect(t_out)

        logger.debug(f"warp_triangle: r_in={r_in} r_out={r_out}")

        t_in_offset = t_in - [r_in[0], r_in[1]]
        t_out_offset = t_out - [r_out[0], r_out[1]]

        img_crop = img[r_in[1]:r_in[1]+r_in[3], r_in[0]:r_in[0]+r_in[2]]
        mask = np.zeros((r_out[3], r_out[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t_out_offset), 255)

        t_in_offset = np.array(t_in_offset, dtype=np.float32)
        t_out_offset = np.array(t_out_offset, dtype=np.float32)

        # skip degenerate triangles
        if t_in.shape[0] != 3 or t_out.shape[0] != 3:
            logger.debug("warp_triangle: degenerate triangle skipped")
            return np.zeros((0,0), dtype=np.uint8), np.zeros((0,0), dtype=np.uint8), (0,0,0,0)


        warp_mat = cv2.getAffineTransform(t_in_offset, t_out_offset)
        warped_patch = cv2.warpAffine(img_crop, warp_mat, (r_out[2], r_out[3]), flags= cv2.INTER_LINEAR, borderMode= cv2.BORDER_REFLECT_101)
        
        return warped_patch, mask, r_out
    

    def morph_triangle(self, img1, img2, img_morph, t1, t2, t, alpha):
        warp1, mask1, r1 = self.warp_triangle(img1, t1, t)
        warp2, mask2, r2 = self.warp_triangle(img2, t2, t)

        blended_patch = cv2.addWeighted(warp1, 1-alpha, warp2, alpha, 0)
        blended_patch = blended_patch.astype(img_morph.dtype)
        logger.debug(f"morph_triangle: blended_patch.shape={getattr(blended_patch,'shape',None)} r1={r1}")
        
        x,y,w,h = r1
        x = max(x, 0)
        y = max(y, 0)
        roi = img_morph[y:y+h, x:x+w]

        mask_out = cv2.merge([mask1, mask1, mask1]) if len(img_morph.shape) == 3 else mask1

        img_morph[y:y+h, x:x+w] = roi * (1-mask_out/255.0) + blended_patch * (mask_out/255.0)


    def get_morphed_face(self, src_img, dst_img, src_points, dst_points, alpha):
        # morphed_img = dst_img.astype(np.uint8)
        morphed_img = np.zeros_like(dst_img, dtype=np.uint8)

        alpha = np.clip(alpha, 0.0, 1.0)
        logger.info(f"get_morphed_face: alpha={alpha} src_points={len(src_points)} dst_points={len(dst_points)}")

        interpolated_points = []
        for i in range(len(src_points)):
            x = (1 - alpha) * src_points[i][0] + alpha * dst_points[i][0]
            y = (1 - alpha) * src_points[i][1] + alpha * dst_points[i][1]
            interpolated_points.append((x, y))

        interpolated_points = np.array(interpolated_points, dtype=np.float32)
        logger.debug(f"get_morphed_face: interpolated {len(interpolated_points)} points")

        h, w = morphed_img.shape[:2]
        # (x, y, w, h) is the expected rect ordering for Subdiv2D
        morphed_img_rect = (0, 0, w, h)

        # build triangulation on the interpolated points
        triangulater = self.triangulator(morphed_img_rect, interpolated_points)
        triangles = triangulater.get_triangles(morphed_img_rect, interpolated_points)
        logger.info(f"get_morphed_face: triangulation returned {len(triangles)} triangles")
        if len(triangles) == 0:
            logger.warning("get_morphed_face: no triangles found; morph will be empty")
            # save debug images so the developer can inspect landmark positions
            try:
                import os
                debug_dir = os.path.join(os.getcwd(), "morph_debug")
                os.makedirs(debug_dir, exist_ok=True)

                # draw landmarks on copies
                try:
                    src_vis = src_img.copy()
                    dst_vis = dst_img.copy()
                    for (x, y) in src_points:
                        cv2.circle(src_vis, (int(x), int(y)), 1, (0, 255, 0), -1)
                    for (x, y) in dst_points:
                        cv2.circle(dst_vis, (int(x), int(y)), 1, (0, 255, 0), -1)

                    interp_vis = morphed_img.copy()
                    for (x, y) in interpolated_points:
                        cv2.circle(interp_vis, (int(x), int(y)), 1, (0, 255, 0), -1)

                    cv2.imwrite(os.path.join(debug_dir, "debug_src_landmarks.png"), src_vis)
                    cv2.imwrite(os.path.join(debug_dir, "debug_dst_landmarks.png"), dst_vis)
                    cv2.imwrite(os.path.join(debug_dir, "debug_interpolated.png"), interp_vis)
                    logger.info(f"Wrote debug landmark images to {debug_dir}")
                except Exception as e:
                    logger.warning(f"Failed writing debug images: {e}")

            except Exception:
                pass

            # Fallback: a coarse whole-face blend so user still sees something morphed
            try:
                blended = self.blend_faces(src_img, dst_img, alpha)
                logger.info("get_morphed_face: returning blended fallback (no triangles)")
                print(f"get_morphed_face: triangulation returned 0 triangles; returning blended fallback")
                return blended
            except Exception as e:
                logger.error(f"Fallback blend failed: {e}")
                return morphed_img

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