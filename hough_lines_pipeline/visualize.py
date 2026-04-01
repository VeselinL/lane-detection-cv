import cv2
import numpy as np
from matplotlib import pyplot as plt
from hough_lines_pipeline.main import detect_lanes
from hough_lines_pipeline.hs_utils import get_roi_mask
from utils.utils import apply_hsl_color_filter
from sliding_window_pipeline.process_frame import process_frame

SLIDING_WINDOW_POINTS = np.float32(
    [
        [565, 330],  # top left
        [750, 330],  # top right
        [1250, 715],  # bot right
        [180, 715],  # bot left
    ]
)


def visualize_pipeline(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: Could not read image from {img_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color_filtered = apply_hsl_color_filter(frame)
    color_filtered_rgb = cv2.cvtColor(color_filtered, cv2.COLOR_GRAY2RGB)
    gray = color_filtered
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    masked_edges = get_roi_mask(edges)
    result = detect_lanes(frame)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # REWROTE: Smaller figsize for better screen fit
    plt.figure(figsize=(12, 8))

    images = [
        (frame_rgb, '1. Original'),
        (color_filtered_rgb, '2. Color Filter'),
        (gray, '3. Grayscale'),
        (blur, '4. Blur'),
        (edges, '5. Canny'),
        (masked_edges, '6. Masked'),
    ]

    for i, (img, title) in enumerate(images):
        plt.subplot(3, 3, i + 1)
        # Use cmap='gray' for single channel images automatically
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title, fontsize=10)
        plt.axis('off')

    # 7. Raw Hough Lines
    plt.subplot(3, 3, 7)
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), 40, 150)
    hough_img = frame_rgb.copy()
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(hough_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(hough_img)
    plt.title('7. Hough Lines', fontsize=10)
    plt.axis('off')

    # 8. Final Result
    plt.subplot(3, 3, 8)
    plt.imshow(result_rgb)
    plt.title('8. Final Result', fontsize=10)
    plt.axis('off')

    # Removed the redundant plot 9 to keep it clean or leave it blank
    plt.tight_layout(pad=1.5)
    plt.savefig('hough_pipeline1.png', dpi=100, bbox_inches='tight')
    plt.show()


def compare_results():
    img_paths = [
        "../test_images/tusimple/test4/10.jpg",
    ]
    source_points = SLIDING_WINDOW_POINTS

    n = len(img_paths)
    plt.figure(figsize=(16, 5 * n))
    previous_lane_state = None

    for index, img_path in enumerate(img_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, _, _, result, previous_lane_state = process_frame(
            frame,
            source_points,
            previous_lane_state,
        )
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.subplot(n, 2, 2 * index + 1)
        plt.imshow(frame_rgb)
        plt.title('1. Original Image')
        plt.axis('off')

        plt.subplot(n, 2, 2 * index + 2)
        plt.imshow(result_rgb)
        plt.title('2. Image with Detected Lanes')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('visualize4.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    compare_results()
