"""V4.6 test helper: synthetic top-down wrist-camera image of an egg carton (dark round cavities, 2 x 3, pitch 47 mm)
whose true slots are shifted (+8, -5) mm from the main-camera estimate, plus a bright egg-shaped distractor."""
import math
import numpy as np
import cv2

def scene(true_off=(0.008, -0.005), pitch=0.047, cam_h=0.16, z=0.022, exp_xy=(-0.33, 0.20), dark=60, bg=175, egg=True,
          cam_dx=-0.04):
    # cam_dx: V4.6.1 - the slot appears in the upper image half (the lower ~42 % is the gripper, excluded)
    W, H = 1280, 720
    K = {"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0, "width": W, "height": H}
    R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)       # camera looks straight down
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [exp_xy[0] + cam_dx, exp_xy[1] - 0.01, z + cam_h]
    img = np.full((H, W, 3), bg, np.uint8)
    def proj(p):
        pc = R.T @ (np.asarray(p) - T[:3, 3])
        return (K["fx"] * pc[0] / pc[2] + K["ppx"], K["fy"] * pc[1] / pc[2] + K["ppy"])
    centres = []
    for r in (-0.5, 0.5):
        for c in (-1, 0, 1):
            cx = exp_xy[0] + true_off[0] + c * pitch
            cy = exp_xy[1] + true_off[1] + (r + 0.5) * pitch
            centres.append((cx, cy))
            poly = np.array([proj((cx + 0.019 * math.cos(t), cy + 0.019 * math.sin(t), z)) for t in np.linspace(0, 2 * math.pi, 60)], np.int32)
            cv2.fillPoly(img, [poly], (dark, dark, dark))
    if egg:   # bright egg-shaped distractor next to the slot
        poly = np.array([proj((exp_xy[0] - 0.03 + 0.027 * math.cos(t), exp_xy[1] - 0.045 + 0.021 * math.sin(t), z)) for t in np.linspace(0, 2 * math.pi, 60)], np.int32)
        cv2.fillPoly(img, [poly], (245, 245, 245))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.RandomState(0).normal(0, 4, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, K, T.tolist(), {"T_flange_cam": np.eye(4).tolist()}, [exp_xy[0], exp_xy[1], z]

