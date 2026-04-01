import cv2
import numpy as np

from process_frame import process_frame
from sliding_window import LANE_WIDTH
from sw_utils import get_dst_points, warp_image
from utils.utils import apply_hsl_color_filter, get_default_hsl_filter_config


VIDEO_PATH = "../test_videos/test2.mp4"
SOURCE_POINTS = np.float32(
        [
            [460, 390], # top left
            [780, 390], # top right
            [1200, 710], # bot right
            [80, 710], # bot left
        ]
    )
CONTROL_WINDOW = "Threshold Controls"
VIEW_WINDOW = "Threshold Tuner"


def _noop(_):
    pass


def _create_trackbars(config):
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 500, 420)
    cv2.createTrackbar("white_l_min", CONTROL_WINDOW, config["white_l_min"], 255, _noop)
    cv2.createTrackbar("white_s_max", CONTROL_WINDOW, config["white_s_max"], 255, _noop)
    cv2.createTrackbar("yellow_h_min", CONTROL_WINDOW, config["yellow_h_min"], 255, _noop)
    cv2.createTrackbar("yellow_h_max", CONTROL_WINDOW, config["yellow_h_max"], 255, _noop)
    cv2.createTrackbar("yellow_l_min", CONTROL_WINDOW, config["yellow_l_min"], 255, _noop)
    cv2.createTrackbar("yellow_s_min", CONTROL_WINDOW, config["yellow_s_min"], 255, _noop)
    cv2.createTrackbar("blur_kernel", CONTROL_WINDOW, config["blur_kernel"], 15, _noop)
    cv2.createTrackbar("open_kernel", CONTROL_WINDOW, config["open_kernel"], 15, _noop)
    cv2.createTrackbar("close_kernel", CONTROL_WINDOW, config["close_kernel"], 15, _noop)


def _read_trackbar_config():
    config = {
        "white_l_min": cv2.getTrackbarPos("white_l_min", CONTROL_WINDOW),
        "white_s_max": cv2.getTrackbarPos("white_s_max", CONTROL_WINDOW),
        "yellow_h_min": cv2.getTrackbarPos("yellow_h_min", CONTROL_WINDOW),
        "yellow_h_max": cv2.getTrackbarPos("yellow_h_max", CONTROL_WINDOW),
        "yellow_l_min": cv2.getTrackbarPos("yellow_l_min", CONTROL_WINDOW),
        "yellow_s_min": cv2.getTrackbarPos("yellow_s_min", CONTROL_WINDOW),
        "blur_kernel": cv2.getTrackbarPos("blur_kernel", CONTROL_WINDOW),
        "open_kernel": cv2.getTrackbarPos("open_kernel", CONTROL_WINDOW),
        "close_kernel": cv2.getTrackbarPos("close_kernel", CONTROL_WINDOW),
    }
    if config["yellow_h_max"] < config["yellow_h_min"]:
        config["yellow_h_max"] = config["yellow_h_min"]
        cv2.setTrackbarPos("yellow_h_max", CONTROL_WINDOW, config["yellow_h_max"])
    return config


def _set_trackbars(config):
    for name, value in config.items():
        cv2.setTrackbarPos(name, CONTROL_WINDOW, int(value))


def _to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _annotate(image, label):
    output = image.copy()
    cv2.putText(output, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return output


def _build_debug_view(frame, mask, warped_mask, result):
    frame_bgr = _annotate(frame, "Source")
    mask_bgr = _annotate(_to_bgr(mask), "Mask")
    warped_bgr = _annotate(_to_bgr(warped_mask), "Warped Mask")
    result_bgr = _annotate(result, "Lane Overlay")

    top = np.hstack((frame_bgr, mask_bgr))
    bottom = np.hstack((warped_bgr, result_bgr))
    grid = np.vstack((top, bottom))

    help_text = "Space: pause/resume | R: reset sliders | Esc: quit"
    cv2.putText(grid, help_text, (15, grid.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return grid


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    default_config = get_default_hsl_filter_config()
    _create_trackbars(default_config)
    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEW_WINDOW, 1400, 900)

    paused = False
    previous_lane_state = None
    previous_config = None
    current_frame = None

    while True:
        if not paused or current_frame is None:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                previous_lane_state = None
                continue
            current_frame = frame

        config = _read_trackbar_config()
        if config != previous_config:
            previous_lane_state = None
            previous_config = config.copy()

        mask = apply_hsl_color_filter(current_frame, config)
        h, w = current_frame.shape[:2]
        left_margin = (w - LANE_WIDTH) // 2
        warped_mask = warp_image(mask, SOURCE_POINTS, get_dst_points(left_margin, LANE_WIDTH, h))
        _, _, _, result, previous_lane_state = process_frame(
            current_frame,
            SOURCE_POINTS,
            previous_lane_state,
            config,
        )

        debug_view = _build_debug_view(current_frame, mask, warped_mask, result)
        cv2.imshow(VIEW_WINDOW, debug_view)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == 27:
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            _set_trackbars(default_config)
            previous_lane_state = None
            previous_config = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
