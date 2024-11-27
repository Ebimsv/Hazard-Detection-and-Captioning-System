import numpy as np
import argparse
from PIL import Image
from utils.video_utils import VideoProcessor
from utils.detection_utils import YOLODetector
from utils.captioning_utils import get_captioner
from utils.state_change_utils import detect_driver_state_change
from collections import defaultdict
import cv2
import os


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Driver hazard detection and captioning."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations/annotations.pkl",
        help="Path to annotations pickle file.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="data/videos/",
        help="Directory containing video files.",
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default="blip_base",
        choices=["blip_base", "instruct_blip", "vit_g"],
        help="Captioning model to use.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations):
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if not os.path.exists(args.video_root):
        raise FileNotFoundError(f"Video root directory not found: {args.video_root}")

    return args


def initialize_results_file(filepath):
    """Initialize the results file with headers."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as file:
        headers = ["ID", "Driver_State_Changed"] + [
            f"Hazard_Track_{i},Hazard_Name_{i}" for i in range(23)
        ]
        file.write(",".join(headers) + "\n")
    return filepath


def process_video(
    video_processor, video_name, yolo_detector, captioner, results_filepath
):
    """Process a single video for hazard detection and captioning."""
    video_path = os.path.join(video_processor.video_root, f"{video_name}.mp4")
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_count = 0
    previous_centroids = []
    median_dists = []
    captioned_tracks = {}
    hazard_history = defaultdict(int)  # Track persistent hazards
    driver_state_flag = False
    driver_state_triggered = (
        False  # Ensure state change is triggered only once per video
    )

    # relevant classes
    relevant_classes = {0, 1, 2, 3, 5, 7, 9, 11, 13, 17, 18, 19, 22, 23}

    while video_stream.isOpened():
        print(f"Processing {video_name}, frame {frame_count}")
        ret, frame_image = video_stream.read()
        if not ret:
            break  # End of video

        # Detect bounding boxes, classes, and track IDs
        bboxes, track_ids, classes, _ = yolo_detector.get_bboxes(frame_image)
        centroids = np.array(
            [(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2) for x1, y1, x2, y2 in bboxes]
        )

        # Skip processing if there are no centroids or no previous centroids
        if centroids.size == 0 or len(previous_centroids) == 0:
            previous_centroids = centroids if centroids.size > 0 else previous_centroids
            frame_count += 1
            continue

        # Calculate movement and detect driver state change
        if (
            not driver_state_triggered
        ):  # Only evaluate if state change hasn't been triggered
            dists = [
                np.min(np.linalg.norm(previous_centroids - centroid, axis=1))
                for centroid in centroids
            ]
            median_dist = np.median(dists)
            median_dists.append(median_dist)

            # Detect driver state change
            if detect_driver_state_change(median_dists):
                driver_state_flag = True
                driver_state_triggered = True  # Lock state change

        # Filter relevant hazards based on object classes
        filtered_indices = [
            i for i, cls in enumerate(classes) if cls in relevant_classes
        ]
        if not filtered_indices:
            frame_count += 1
            continue

        filtered_centroids = centroids[filtered_indices]
        filtered_track_ids = [track_ids[i] for i in filtered_indices]
        filtered_bboxes = [bboxes[i] for i in filtered_indices]

        # Identify the closest hazard to the center of the screen
        image_center = np.array([frame_image.shape[1] / 2, frame_image.shape[0] / 2])
        potential_hazard_dists = np.linalg.norm(
            filtered_centroids - image_center, axis=1
        )
        probable_hazard_index = np.argmin(potential_hazard_dists)
        hazard_track = filtered_track_ids[probable_hazard_index]
        hazard_bbox = filtered_bboxes[probable_hazard_index]

        # Update hazard history for temporal consistency
        hazard_history[hazard_track] += 1
        if (
            hazard_history[hazard_track] < 3
        ):  # Ignore hazards appearing for fewer than 3 frames
            frame_count += 1
            continue

        # Generate or reuse hazard captions
        if hazard_track not in captioned_tracks:
            hazard_chip = Image.fromarray(
                cv2.cvtColor(
                    frame_image[
                        int(hazard_bbox[1]) : int(hazard_bbox[3]),
                        int(hazard_bbox[0]) : int(hazard_bbox[2]),
                    ],
                    cv2.COLOR_BGR2RGB,
                )
            )
            hazard_caption = captioner.get_caption(hazard_chip).replace(",", " ")
            captioned_tracks[hazard_track] = hazard_caption
        else:
            hazard_caption = captioned_tracks[hazard_track]

        # Prepare hazard tracks and names for CSV
        hazard_tracks = [str(hazard_track)]
        hazard_names = [hazard_caption]

        # Fill up to 22 hazard slots with empty strings
        hazard_tracks += [""] * (22 - len(hazard_tracks))
        hazard_names += [""] * (22 - len(hazard_names))

        # Write results to CSV
        with open(results_filepath, "a") as results_file:
            results_line = [f"{video_name}_{frame_count}", str(driver_state_flag)]
            for track, name in zip(hazard_tracks[:22], hazard_names[:22]):
                results_line.extend([track, name])
            results_file.write(",".join(map(str, results_line)) + "\n")

        # Update centroids and frame count
        previous_centroids = centroids
        frame_count += 1

    video_stream.release()


def main():
    args = parse_arguments()
    results_filepath = initialize_results_file("results/results.csv")

    # Initialize utilities
    video_processor = VideoProcessor(args.annotations, args.video_root)
    yolo_detector = YOLODetector()
    captioner = get_captioner(args.caption_model)

    # Hardcoded video name
    video_name = "video_0001"

    if video_name in video_processor.annotations:
        process_video(
            video_processor, video_name, yolo_detector, captioner, results_filepath
        )
    else:
        print(f"Video {video_name} not found in annotations.")


# Computing operations on all videos
# def main():
#     args = parse_arguments()
#     results_filepath = initialize_results_file("results/results.csv")

#     # Initialize utilities
#     video_processor = VideoProcessor(args.annotations, args.video_root)
#     yolo_detector = YOLODetector()
#     captioner = get_captioner(args.caption_model)

#     # Process each video
#     for video_name in sorted(video_processor.annotations.keys()):
#         process_video(
#             video_processor, video_name, yolo_detector, captioner, results_filepath
#         )


if __name__ == "__main__":
    main()
