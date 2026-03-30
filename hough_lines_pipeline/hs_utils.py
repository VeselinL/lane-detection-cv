import cv2
import numpy as np

def get_roi_mask(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    bottom_left = (int(width * 0.10), height)
    top_left = (int(width * 0.30), int(height * 0.40))
    top_right = (int(width * 0.70), int(height * 0.40))
    bottom_right = (int(width * 0.90), height)

    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None

    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.60)

    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except (OverflowError, ZeroDivisionError):
        return None

    return np.array([x1, y1, x2, y2])