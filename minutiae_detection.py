import cv2 
import numpy as np
from typing import NamedTuple

# similar to LEADER Minutia
class Minutia(NamedTuple):
    x: float
    y: float
    direction: float

def offset_minutiae(minutiae, roi_coords, scale):
    y0, x0 = np.asarray(roi_coords) * scale

    return [m._replace(x=m.x + x0, y=m.y + y0) for m in minutiae]

# detect minutiae on the annotated image
def detect_annotated_minutiae(imageset, theta=None, min_area=10, max_area=5000, blur_ksize=3):
    debug = {}

    img_bgr = imageset.normalisedROICA

    # 1) (Optional) denoise a bit
    if blur_ksize and blur_ksize > 1:
        img_bgr_blur = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
    else:
        img_bgr_blur = img_bgr.copy()
    debug["blur"] = img_bgr_blur

    # Convert to HSV for color thresholding
    hsv = cv2.cvtColor(img_bgr_blur, cv2.COLOR_BGR2HSV)
    debug["hsv"] = hsv

    # Threshold for 'pink' - Tuned for magenta shades
    lower1 = np.array([145, 160, 200])   # H, S, V
    upper1 = np.array([165, 255, 255])

    # Some pinks may drift lower (more red-ish). Add a second band if needed:
    lower2 = np.array([145, 160, 200])
    upper2 = np.array([165, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    debug["mask_raw"] = mask

    # Morphological cleanup - remove speckles, fill dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    debug["mask_clean"] = mask

    # Find connected components / contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # center via moments
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        t = theta[cy, cx]
        minutia = Minutia(x=cx, y=cy, direction=t)
        centers.append(minutia)

    # add roi offset to minutiae.
    centers = offset_minutiae(centers, imageset.roiCoords, imageset.median_scale)

    # Optional debug overlay
    overlay = img_bgr.copy()
    for c in centers:
        center = (int(round(c.x)), int(round(c.y)))
        cv2.circle(overlay, center, 6, (0, 255, 0), 2)
        cv2.circle(overlay, center, 2, (0, 255, 0), -1)
    debug["overlay"] = overlay

    return centers, mask, debug

def detect_minutiae_automatic(imageset, detector):
    detected = detector.run(imageset.normalisedROI)

    # adjust minutiae with roi offset
    detected = offset_minutiae(detected, imageset.roiCoords, imageset.median_scale)

    return detected
