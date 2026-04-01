import cv2
import numpy as np

from hough_lines_pipeline.hs_utils import get_roi_mask, make_coordinates
from utils.utils import apply_hsl_color_filter


prev_left_fit = None
prev_right_fit = None


def average_slope_intercept(image, lines):
    global prev_left_fit, prev_right_fit

    alpha = 0.2
    left_fits = []
    right_fits = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue

                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]

                if -3.0 < slope < -0.3:
                    left_fits.append((slope, intercept))
                elif 0.3 < slope < 3.0:
                    right_fits.append((slope, intercept))

    current_left_avg = np.average(left_fits, axis=0) if left_fits else None
    current_right_avg = np.average(right_fits, axis=0) if right_fits else None

    if current_left_avg is not None:
        if prev_left_fit is None:
            prev_left_fit = current_left_avg
        else:
            prev_left_fit = (current_left_avg * alpha) + (prev_left_fit * (1 - alpha))

    if current_right_avg is not None:
        if prev_right_fit is None:
            prev_right_fit = current_right_avg
        else:
            prev_right_fit = (current_right_avg * alpha) + (prev_right_fit * (1 - alpha))

    final_lines = []
    left_line = make_coordinates(image, prev_left_fit)
    right_line = make_coordinates(image, prev_right_fit)

    if left_line is not None:
        final_lines.append(left_line)
    if right_line is not None:
        final_lines.append(right_line)

    return np.array(final_lines)


def detect_lanes(frame):
    color_filtered = apply_hsl_color_filter(frame)
    blur = cv2.GaussianBlur(color_filtered, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    masked_edges = get_roi_mask(edges)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=150,
    )

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = np.zeros_like(frame)

    if averaged_lines is not None:
        for line in averaged_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 12)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def process_frame(frame):
    result = detect_lanes(frame)
    return result


def process_image(input_path, output_path=None):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image from {input_path}")
        return

    result = process_frame(frame)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved to {output_path}")
    else:
        cv2.imshow("Lane Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    process_image("../test_images/tusimple/test1/10.jpg")


if __name__ == "__main__":
    main()
