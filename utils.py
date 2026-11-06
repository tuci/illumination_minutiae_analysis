import numpy as np
import cv2

def compute_tp_fp_fn(annotated, detected, dist_thresh=10.0, offset=(18, 18)):
    """
    annotated: list/array of (x,y) ground-truth points
    detected: dict, values contain {"x":..., "y":...}
    dist_thresh: max distance to consider a match
    offset: (ox, oy) added to detected coords (use (0,0) if none)

    Returns:
      TP: list of (gt_idx, det_idx)
      FP: list of det_idx
      FN: list of gt_idx
      det_pts: list of (x,y) (for visualization)
    """
    # normalize annotated
    annotated_pts = [(float(p.x), float(p.y)) for p in annotated]

    ox, oy = offset if isinstance(offset, (tuple, list)) else (offset, offset)

    # keep a stable order for detected points
    det_pts = [(float(v.x) + ox, float(v.y) + oy) for v in detected]

    if len(annotated_pts) == 0:
        TP = []
        FP = list(range(len(det_pts)))
        FN = []
        return TP, FP, FN

    if len(det_pts) == 0:
        TP = []
        FP = []
        FN = list(range(len(annotated_pts)))
        return TP, FP, FN

    # Build candidate matches
    candidates = []
    for i, (gx, gy) in enumerate(annotated_pts):
        for j, (dx, dy) in enumerate(det_pts):
            d = float(np.hypot(gx - dx, gy - dy))
            if d <= dist_thresh:
                candidates.append((d, i, j))

    # Closest-first matching (one-to-one)
    candidates.sort(key=lambda x: x[0])

    gt_used, det_used = set(), set()
    TP = []
    for d, i, j in candidates:
        if i in gt_used or j in det_used:
            continue
        gt_used.add(i)
        det_used.add(j)
        TP.append((i, j, annotated[i], detected[j]))

    FN = [(i, annotated[i]) for i in range(len(annotated_pts)) if i not in gt_used]
    FP = [(j, detected[j]) for j in range(len(det_pts)) if j not in det_used]

    return TP, FP, FN

def visualize_tp_fp_fn_with_gt(image, annotated, detected, TP, FP, FN,  
    offset=(18, 18), draw_lines=True, draw_orientation=True, line_length=20, gt_radius=3, 
    tp_radius=4, fp_radius=6, fn_halfsize=5, gt_thickness=1, fp_thickness=2, fn_thickness=2):
    """
    Draws:
      - All annotated GT points in black (thin circle)
      - TP in dark green (filled circle) + optional line to detected
      - FP in red (circle + X)
      - FN in blue (square)

    Returns: BGR image with overlays.
    """

    # Ensure BGR image for drawing
    if image is None:
        raise ValueError("image is None")
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    # Normalize coords
    ann = [(float(p.x), float(p.y)) for p in annotated]
    ox, oy = offset if isinstance(offset, (tuple, list)) else (offset, offset)
    det_pts = [(float(d.x) + ox, float(d.y) + oy, d.direction) for d in detected]

    # Colors (OpenCV BGR) tuned for light/beige backgrounds
    C_GT = (0, 0, 0)         # black
    C_TP = (59, 200, 255)   #(0, 160, 0)       # dark green
    C_FP = (0, 0, 200)       # red/crimson
    C_FN = (200, 0, 0)       # blue
    C_LINE = (59, 200, 255)  #(0, 160, 0)     # same as TP

    # 1) Draw all GT (context) first
    for (x, y) in ann:
        cv2.circle(out, (int(round(x)), int(round(y))), gt_radius, C_GT, gt_thickness, lineType=cv2.LINE_AA)

    # 2) Draw FN (missed GT) as blue squares (on top of GT)
    for i, _ in FN:
        if i < 0 or i >= len(ann):
            continue
        x, y = ann[i]
        x, y = int(round(x)), int(round(y))
        cv2.rectangle(
            out,
            (x - fn_halfsize, y - fn_halfsize),
            (x + fn_halfsize, y + fn_halfsize),
            C_FN,
            fn_thickness,
            lineType=cv2.LINE_AA
        )

    # 3) Draw FP (extra detections) as red circle + X
    for j, _ in FP:
        if j < 0 or j >= len(det_pts):
            continue
        x, y, theta = det_pts[j]
        x, y = int(round(x)), int(round(y))

        # Circle
        cv2.circle(out, (x, y), fp_radius, C_FP, fp_thickness, lineType=cv2.LINE_AA)

        # X marker
        s = max(4, fp_radius)
        cv2.line(out, (x - s, y - s), (x + s, y + s), C_FP, max(1, fp_thickness - 1), lineType=cv2.LINE_AA)
        cv2.line(out, (x - s, y + s), (x + s, y - s), C_FP, max(1, fp_thickness - 1), lineType=cv2.LINE_AA)

        # orientation
        if draw_orientation:
            x_ori = int(x + line_length * np.cos(theta))
            y_ori = int(y - line_length * np.sin(theta))
            cv2.arrowedLine(out, (x, y), (x_ori, y_ori), C_FP, 2, tipLength=0.2)

    # 4) Draw TP as filled dark-green circle (and optional line to detection)
    for (i, j, _, _) in TP:
        if i < 0 or i >= len(ann) or j < 0 or j >= len(det_pts):
            continue

        xg, yg = ann[i]
        xd, yd, theta = det_pts[j]
        xg, yg = int(round(xg)), int(round(yg))
        xd, yd = int(round(xd)), int(round(yd))

        if draw_lines:
            cv2.line(out, (xg, yg), (xd, yd), C_LINE, 1, lineType=cv2.LINE_AA)

        cv2.circle(out, (xg, yg), tp_radius, C_TP, -1, lineType=cv2.LINE_AA)

        if draw_orientation:
            x_ori = int(xg + line_length * np.cos(theta))
            y_ori = int(yg - line_length * np.sin(theta))
            cv2.arrowedLine(out, (xg, yg), (x_ori, y_ori), C_TP, 2, tipLength=0.2)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def visualise_minutiae(image, detected, offset=(18, 18), draw_lines=True,
                       gt_radius=3, gt_thickness=1):
    # Ensure BGR image for drawing
    if image is None:
        raise ValueError("image is None")
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    # Normalize coords
    oy, ox = offset
    det_pts = [(float(d.x) + ox, float(d.y) + oy) for d in detected]

    # 1) Draw all GT (context) first
    C_GT = (200, 0, 0)         # black
    for (x, y) in det_pts:
        cv2.circle(out, (int(round(x)), int(round(y))), gt_radius, C_GT, gt_thickness, lineType=cv2.LINE_AA)

    return out
