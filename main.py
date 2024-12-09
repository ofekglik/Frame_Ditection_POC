import cv2
from utils import ensure_dir, save_frame
from frame_processor import calculate_sharpness, detect_changes, calculate_composite_sharpness
import os


def main(grid_search=False):
    video_path = ''
    output_dir = ''
    top_frames_base_dir = ''

    num_top_frames = 3

    if grid_search:
        weight_combinations = [
            (0.5, 0.3, 0.2),
            (0.3, 0.4, 0.3),
            (0.1, 0.5, 0.4),
            (0.4, 0.3, 0.3),
        ]
    else:
        weight_combinations = [(0.5, 0.3, 0.2)]

    for weights in weight_combinations:
        top_frames_dir = os.path.join(top_frames_base_dir, f'weights_{weights[0]}_{weights[1]}_{weights[2]}')
        ensure_dir(top_frames_dir)

        ensure_dir(output_dir)

        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()

        threshold = 25
        num_of_diff_pixels = 50000
        frame_id = 0
        sharpness_scores = {}

        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            changes = detect_changes(prev_frame, frame, threshold, num_of_diff_pixels)
            for roi, x, y, w, h in changes:
                sharpness_score = calculate_composite_sharpness(roi, weights)
                sharpness_scores[frame_id] = sharpness_score
                save_frame(roi, output_dir, f'detected_frame_{frame_id}.jpg')

            prev_frame = frame.copy()
            frame_id += 1

        cap.release()

        top_frames = sorted(sharpness_scores.items(), key=lambda x: x[1], reverse=True)[:num_top_frames]
        for frame_id, _ in top_frames:
            frame_path = f'detected_frame_{frame_id}.jpg'
            save_frame(cv2.imread(os.path.join(output_dir, frame_path)), top_frames_dir,
                       f'top_frame_{frame_id}.jpg')

        print(f"Top frames for weights {weights} saved successfully in {top_frames_dir}.")


if __name__ == "__main__":
    gs = False
    main(grid_search=gs)
