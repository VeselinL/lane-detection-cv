import numpy as np
import cv2
import sys

def get_histogram(image):
    h = image.shape[0]
    w = image.shape[1]
    mid = h // 2
    lower_half = image[mid:, :]
    np.set_printoptions(threshold=sys.maxsize)
    histogram = np.sum(lower_half, axis=0)
    horizontal_mid = w // 2

    leftx_base = np.argmax(histogram[:horizontal_mid])
    rightx_base = np.argmax(histogram[horizontal_mid:]) + horizontal_mid

    return leftx_base, rightx_base

def warp_image(img, src_pts, dst_pts):
    width, height = img.shape[1], img.shape[0]
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (width,height))
    return warped

def draw_circles(img, pts):
    pts = pts.astype(int)
    for pt in pts:
        cv2.circle(img, tuple(pt), 5, (0,0,255), -1)
    return img

def draw_points_and_roi(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    debug = image.copy()
    cv2.polylines(debug, [points.astype(int).reshape(-1, 1, 2)], True, (0, 0, 255), 3)

    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, points.astype(int)):
        cv2.circle(debug, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(
            debug,
            label,
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return debug

def get_dst_points(left_margin, lane_width, h):
    return np.float32(
        [
            [left_margin, 0],
            [left_margin + lane_width, 0],
            [left_margin + lane_width, h - 1],
            [left_margin, h - 1],
        ]
    )