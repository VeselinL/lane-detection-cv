import cv2
import numpy as np
from process_frame import process_frame

video_test_one = np.float32(
        [
            [550, 570], # top left
            [700, 570], # top right
            [900, 710], # bot right
            [350, 710], # bot left
        ]
    )
video_test_two = np.float32(
        [
            [460, 390], # top left
            [780, 390], # top right
            [1200, 710], # bot right
            [80, 710], # bot left
        ]
    )
video_test_four = np.float32(
        [
            [540, 390], # top left
            [780, 390], # top right
            [1200, 710], # bot right
            [20, 710], # bot left
        ]
)
video_test_three = np.float32(
        [
            [620, 500], # top left
            [870, 500], # top right
            [1150, 710], # bot right
            [300, 710], # bot left
        ]
)
def process_video(src_pts, filter_config=None, output_path=None):
    cap = cv2.VideoCapture("../test_videos/test2.mp4")
    cv2.namedWindow("Final result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Final result", 800, 500)
    cv2.namedWindow("Windows", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Windows", 800, 500)
    cv2.namedWindow("Warped image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warped image", 800, 500)
    cv2.namedWindow("Drawn points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drawn points", 800, 500)


    out = None
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    previous_lane_state = None
    while True:
        success, frame = cap.read()
        if not success:
            break
        circles, warped, windows, res, previous_lane_state = process_frame(
            frame,
            src_pts,
            previous_lane_state,
            filter_config,
        )

        cv2.imshow("Final result", res)
        cv2.imshow("Warped image", warped)
        cv2.imshow("Windows", windows)
        cv2.imshow("Drawn points", circles )
        if out:
            out.write(res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(10) == 27:
            break
    cap.release()
    if out:
        out.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()
process_video(video_test_two, output_path="output/videos/test2.mp4")
