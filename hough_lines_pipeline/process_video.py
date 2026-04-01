import cv2

from hough_lines_pipeline.process_frame import process_frame


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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        cv2.imshow("Lane Detection", processed)

        if out:
            out.write(processed)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()


def main():
    process_video("../test_videos/test4.mp4", "output/videos/test4.mp4")


if __name__ == "__main__":
    main()
