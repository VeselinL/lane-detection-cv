import cv2
import numpy as np


DEFAULT_HSL_FILTER_CONFIG = {
    "white_l_min": 180,
    "white_s_max": 255,
    "yellow_h_min": 10,
    "yellow_h_max": 40,
    "yellow_l_min": 0,
    "yellow_s_min": 80,
    "blur_kernel": 0,
    "open_kernel": 0,
    "close_kernel": 0,
}


def get_default_hsl_filter_config():
    return DEFAULT_HSL_FILTER_CONFIG.copy()


def _normalized_kernel_size(value):
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def _merge_filter_config(config):
    merged = get_default_hsl_filter_config()
    if config:
        merged.update(config)

    merged["white_l_min"] = int(np.clip(merged["white_l_min"], 0, 255))
    merged["white_s_max"] = int(np.clip(merged["white_s_max"], 0, 255))
    merged["yellow_h_min"] = int(np.clip(merged["yellow_h_min"], 0, 255))
    merged["yellow_h_max"] = int(np.clip(merged["yellow_h_max"], merged["yellow_h_min"], 255))
    merged["yellow_l_min"] = int(np.clip(merged["yellow_l_min"], 0, 255))
    merged["yellow_s_min"] = int(np.clip(merged["yellow_s_min"], 0, 255))
    merged["blur_kernel"] = max(0, int(merged["blur_kernel"]))
    merged["open_kernel"] = max(0, int(merged["open_kernel"]))
    merged["close_kernel"] = max(0, int(merged["close_kernel"]))

    return merged


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

def apply_hsl_color_filter(frame, config=None):
    params = _merge_filter_config(config)

    blur_kernel = _normalized_kernel_size(params["blur_kernel"])
    filtered_frame = frame
    if blur_kernel > 1:
        filtered_frame = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)

    hls = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, params["white_l_min"], 0], dtype=np.uint8)
    upper_white = np.array([255, 255, params["white_s_max"]], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array(
        [params["yellow_h_min"], params["yellow_l_min"], params["yellow_s_min"]],
        dtype=np.uint8,
    )
    upper_yellow = np.array([params["yellow_h_max"], 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    open_kernel = _normalized_kernel_size(params["open_kernel"])
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    close_kernel = _normalized_kernel_size(params["close_kernel"])
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    return combined_mask

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
