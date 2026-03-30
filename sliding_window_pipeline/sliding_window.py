import cv2
import numpy as np
from sw_utils import get_dst_points, get_histogram

# slides windows across the warped image, finds the line pixels and fills the lane
def sliding_windows(image, source, source_points):
    h, w = image.shape[:2]
    windows = image.copy()
    n_windows = 10
    window_height = h // n_windows

    leftx_base, rightx_base = get_histogram(image)
    margin = 50
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    for window in range(n_windows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin

        # draw rectangles
        cv2.rectangle(windows, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
        cv2.rectangle(windows, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

        # y bounds — start from bottom, move up
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        # x bounds around current center
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin

        # find nonzero pixels in each window
        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        # recenter for next window, if pixels found
        if len(good_left) > minpix:
            leftx_base = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_base = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # draw on warped
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # fill the lane
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))


    # draw lines
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
    cv2.polylines(color_warp, pts_left, isClosed=False, color=(255, 0, 0), thickness=25)
    cv2.polylines(color_warp, pts_right, isClosed=False, color=(0, 0, 255), thickness=25)

    # inverse homography
    h, w = image.shape[:2]
    lane_width = 575
    left_margin = (w - lane_width) // 2
    M_inv = cv2.getPerspectiveTransform(get_dst_points(left_margin, lane_width, h), source_points)
    unwarped = cv2.warpPerspective(color_warp, M_inv, (source.shape[1], source.shape[0]))

    # combine the source and the processed image
    result = cv2.addWeighted(source, 1, unwarped, 0.7, 0.3)

    return result, windows