import numpy as np
import cv2
from utils.utils import apply_hsl_color_filter
from sliding_window import sliding_windows
from sw_utils import warp_image, draw_circles, get_dst_points

tusimple_test1_pts = np.float32(
        [
            [565, 330], # top left
            [750, 330], # top right
            [1250, 715], # bot right
            [180, 715], # bot left
        ]
    )

def process_frame(frame, src_pts):

    h, w = frame.shape[:2]
    lane_width = 575
    left_margin = (w - lane_width) // 2
    filtered = apply_hsl_color_filter(frame)
    warped = warp_image(filtered, src_pts, get_dst_points(left_margin, lane_width, h))
    circles = draw_circles(frame.copy(), src_pts)
    result, windows = sliding_windows(warped, frame, src_pts)
    return circles, warped, windows, result

def main():
    source = cv2.imread(f"../test_images/tusimple/test1/10.jpg")
    if source is None:
        print(f"could not read image 10.jpg")
        return
    circles, warped, windows, result = process_frame(source, tusimple_test1_pts)

    cv2.namedWindow("Final result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Final result", 800, 500)
    cv2.namedWindow("Sliding windows", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sliding windows", 800, 500)
    cv2.namedWindow("Warped image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warped image", 800, 500)
    cv2.namedWindow("Drawn points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drawn points", 800, 500)

    cv2.imshow("Final result", result)
    cv2.imshow("Sliding windows", windows)
    cv2.imshow("Warped image", warped)
    cv2.imshow("Drawn points", circles)

    #cv2.imwrite(f"output/images/tusimple/test1/10.jpg", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()