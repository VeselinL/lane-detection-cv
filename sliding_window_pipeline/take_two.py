import cv2
import numpy as np
from sw_utils import get_dst_points, get_histogram
from utils.utils import apply_hsl_color_filter
from sw_utils import warp_image
from process_frame import process_frame
tusimple_test1_pts = np.float32(
        [
            [565, 330], # top left
            [750, 330], # top right
            [1250, 715], # bot right
            [180, 715], # bot left
        ]
    )
def sliding_window(image, src_pts):
    h, w = image.shape[:2]
    lane_width = 575
    left_margin = (w - lane_width) // 2

    image = apply_hsl_color_filter(image)
    image = warp_image(image, tusimple_test1_pts, get_dst_points(left_margin, lane_width, h))

    n_windows = 10
    window_height = h // n_windows

    leftx_base, rightx_base = get_histogram(image)

    margin = 50
    minpix = 50

    nonzero = np.nonzero(image)
    nonzerox = nonzero[1]
    nonzeroy = nonzero[0]

    left_lane_inds = []
    right_lane_inds = []
    for window in range(n_windows):
        winy_low = h - (window + 1) * window_height
        winy_high = h - window * window_height

        leftx_low = leftx_base - margin
        leftx_high = leftx_base + margin

        rightx_low = rightx_base - margin
        rightx_high = rightx_base + margin

        cv2.rectangle(image, (leftx_low, winy_low), (leftx_high, winy_high), (255, 255, 255), 2)
        cv2.rectangle(image, (rightx_low, winy_low), (rightx_high, winy_high), (255, 255, 255), 2)

        good_left = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= leftx_low) & (nonzerox < leftx_high)).nonzero()[0]
        good_right = ((nonzeroy >= winy_low) & (nonzeroy < winy_high) & (nonzerox >= rightx_low) & (nonzerox < rightx_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        leftx = nonzerox[good_left]
        lefty = nonzeroy[good_left]
        rightx = nonzerox[good_right]
        righty = nonzeroy[good_right]

        if len(good_left) > minpix:
            leftx_base = int(np.mean(leftx))
        if len(good_right) > minpix:
            rightx_base = (np.mean(rightx))

        #print(rightx.shape)
        #left_fit = np.polyfit(lefty, leftx, 2)
        #right_fit = np.polyfit(righty, rightx, 2)

        #cv2.imshow("Image", image)
        #cv2.waitKey(200)





source = cv2.imread(f"../test_images/tusimple/test1/10.jpg")
sliding_window(source, tusimple_test1_pts)
