import cv2
from utils import ensure_dir, save_frame
from frame_processor import calculate_sharpness, detect_changes, calculate_composite_sharpness
import os


def main(grid_search=False):
    # Paths
    video_path = '/Users/ofekglik/PycharmProjects/automunch/Videos/video1.mp4'
    output_dir = '/Users/ofekglik/PycharmProjects/automunch/frames/'
    top_frames_base_dir = '/Users/ofekglik/PycharmProjects/automunch/top_frames/'

    num_top_frames = 3

    # Grid search parameters for weights
    if grid_search:
        weight_combinations = [
            (0.5, 0.3, 0.2),
            (0.3, 0.4, 0.3),
            (0.1, 0.5, 0.4),
            (0.4, 0.3, 0.3),
        ]
    else:
        # Specific weights
        weight_combinations = [(0.5, 0.3, 0.2)]

    for weights in weight_combinations:
        # Specific directory for each set of weights
        top_frames_dir = os.path.join(top_frames_base_dir, f'weights_{weights[0]}_{weights[1]}_{weights[2]}')
        ensure_dir(top_frames_dir)

        # Ensure directories exist
        ensure_dir(output_dir)

        # Open video
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

        # Sort frames by sharpness and save the top frames (num_top_frames)
        top_frames = sorted(sharpness_scores.items(), key=lambda x: x[1], reverse=True)[:num_top_frames]
        for frame_id, _ in top_frames:
            frame_path = f'detected_frame_{frame_id}.jpg'
            save_frame(cv2.imread(os.path.join(output_dir, frame_path)), top_frames_dir,
                       f'top_frame_{frame_id}.jpg')

        print(f"Top frames for weights {weights} saved successfully in {top_frames_dir}.")


if __name__ == "__main__":
    gs = False
    main(grid_search=gs)
