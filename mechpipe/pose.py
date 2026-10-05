"""A-pose pose reference: an OpenPose-style skeleton (COCO-18 keypoints, standard colors) with mech
proportions (wide shoulders, small head, legs apart).

Given as image 2 of the A-pose edit, it makes 2511 actually re-pose mechs drawn in 3/4 or crouched stances;
with text alone (any prompt, draft or final mode) such a mech comes back unchanged. A human-proportioned
skeleton turned the mech into a slim humanoid, so the shoulders here are deliberately wide.
"""
from PIL import Image, ImageDraw

# keypoint -> (x, y) as a fraction of the canvas; viewer facing, so the mech's right side is image left
KEYPOINTS = {
    0: (.5, .13), 1: (.5, .2),                                     # nose, neck
    2: (.28, .2), 3: (.22, .38), 4: (.2, .53),                     # right shoulder, elbow, wrist
    5: (.72, .2), 6: (.78, .38), 7: (.8, .53),                     # left shoulder, elbow, wrist
    8: (.4, .5), 9: (.38, .7), 10: (.37, .9),                      # right hip, knee, ankle
    11: (.6, .5), 12: (.62, .7), 13: (.63, .9),                    # left hip, knee, ankle
    14: (.485, .12), 15: (.515, .12), 16: (.47, .13), 17: (.53, .13),  # eyes, ears
}
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def apose_skeleton(size: tuple[int, int] = (864, 1152)) -> Image.Image:
    w, h = size
    line, dot = max(4, round(w / 62)), max(3, round(w / 96))
    im = Image.new("RGB", size, "black")
    d = ImageDraw.Draw(im)
    pt = {k: (x * w, y * h) for k, (x, y) in KEYPOINTS.items()}
    for i, (a, b) in enumerate(LIMBS):
        d.line([pt[a], pt[b]], fill=COLORS[i], width=line)
    for k, (x, y) in pt.items():
        d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=COLORS[k])
    return im
