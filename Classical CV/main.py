import os
import cv2
import numpy as np

prev_left_fit = None
prev_right_fit = None


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


def apply_hsl_color_filter(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(frame, frame, mask=combined_mask)


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
    gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    masked_edges = get_roi_mask(edges)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=150
    )

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = np.zeros_like(frame)

    if averaged_lines is not None:
        for line in averaged_lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def process_image(input_path, output_path=None):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image from {input_path}")
        return

    result = detect_lanes(frame)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved to {output_path}")
    else:
        cv2.imshow('Lane Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(input_path, output_path=None):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = detect_lanes(frame)
        cv2.imshow('Lane Detection', processed)

        if out:
            out.write(processed)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
        process_video('videos/test2.mp4', 'videos/output/test2.mp4')
