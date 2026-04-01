import cv2
import numpy as np
from matplotlib import pyplot as plt

from sliding_window_pipeline.process_frame import tusimple_test1_pts
from sliding_window_pipeline.sliding_window import HISTOGRAM_SIDE_CROP_RATIO, LANE_WIDTH
from sliding_window_pipeline.sw_utils import (
    draw_points_and_roi,
    get_dst_points,
    get_histogram,
    warp_image,
)
from utils.utils import apply_hsl_color_filter


def _build_sliding_window_debug(warped_binary):
    h, w = warped_binary.shape[:2]
    windows = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
    n_windows = 10
    window_height = h // n_windows
    margin = 50
    minpix = 50

    leftx_base, rightx_base = get_histogram(
        warped_binary,
        side_crop_ratio=HISTOGRAM_SIDE_CROP_RATIO,
    )

    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin

        cv2.rectangle(
            windows,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            windows,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (255, 255, 255),
            2,
        )

        good_left = (
            (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            )
            .nonzero()[0]
        )
        good_right = (
            (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            )
            .nonzero()[0]
        )

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_base = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_base = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    left_fit = None
    right_fit = None
    if len(left_lane_inds) >= 3:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
        windows[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    if len(right_lane_inds) >= 3:
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
        windows[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ploty = np.linspace(0, h - 1, h)
    lane_markings = np.zeros((h, w, 3), dtype=np.uint8)

    left_fitx = None
    right_fitx = None
    if left_fit is not None:
        left_fitx = np.clip(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], 0, w - 1)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(lane_markings, pts_left, isClosed=False, color=(255, 0, 0), thickness=25)
    if right_fit is not None:
        right_fitx = np.clip(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], 0, w - 1)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(lane_markings, pts_right, isClosed=False, color=(0, 0, 255), thickness=25)

    if left_fitx is not None and right_fitx is not None:
        pts_left_fill = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right_fill = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left_fill, pts_right_fill))
        cv2.fillPoly(lane_markings, np.int32([pts]), (0, 255, 0))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(lane_markings, pts_left, isClosed=False, color=(255, 0, 0), thickness=25)
        cv2.polylines(lane_markings, pts_right, isClosed=False, color=(0, 0, 255), thickness=25)

    return windows, lane_markings


def visualize_pipeline():
    img_path = "../test_images/tusimple/test4/10.jpg"
    source_points = tusimple_test1_pts

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: Could not read image from {img_path}")
        return

    h, w = frame.shape[:2]
    left_margin = (w - LANE_WIDTH) // 2
    dst_points = get_dst_points(left_margin, LANE_WIDTH, h)

    roi_debug = draw_points_and_roi(frame.copy(), source_points)
    thresholded = apply_hsl_color_filter(frame)
    warped_binary = warp_image(thresholded, source_points, dst_points)
    windows_debug, lane_markings = _build_sliding_window_debug(warped_binary)

    inverse_matrix = cv2.getPerspectiveTransform(dst_points, source_points)
    unwarped_lane = cv2.warpPerspective(lane_markings, inverse_matrix, (w, h))
    final_result = cv2.addWeighted(frame, 1, unwarped_lane, 0.7, 0.3)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_debug_rgb = cv2.cvtColor(roi_debug, cv2.COLOR_BGR2RGB)
    thresholded_rgb = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    warped_binary_rgb = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2RGB)
    windows_debug_rgb = cv2.cvtColor(windows_debug, cv2.COLOR_BGR2RGB)
    lane_markings_rgb = cv2.cvtColor(lane_markings, cv2.COLOR_BGR2RGB)
    unwarped_lane_rgb = cv2.cvtColor(unwarped_lane, cv2.COLOR_BGR2RGB)
    final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    images = [
        (frame_rgb, "1. Original Image"),
        (roi_debug_rgb, "2. Source ROI"),
        (thresholded_rgb, "3. Threshold Mask"),
        (warped_binary_rgb, "4. Bird's-Eye View"),
        (windows_debug_rgb, "5. Sliding Windows"),
        (lane_markings_rgb, "6. Lane Markings"),
        (unwarped_lane_rgb, "7. Unwarped Lane"),
        (final_result_rgb, "8. Final Overlay"),
    ]

    for ax, (image, title) in zip(axes.flat, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis("off")

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(hspace=0.08, wspace=0.03)
    fig.savefig("sliding_window_pipeline4.png", dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    visualize_pipeline()
