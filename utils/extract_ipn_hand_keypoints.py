import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand keypoint sequence features for IPN Hand videos."
    )
    parser.add_argument("--raw_dir", default="dataset/IPN Hand")
    parser.add_argument("--prepared_dir", default="dataset/ipn_hand")
    parser.add_argument("--output_name", default="features_keypoint")
    parser.add_argument(
        "--with_delta",
        action="store_true",
        help="also save coordinate delta features to features_keypoint_delta",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="optional video names to process",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is not installed. Install it in the asrf env first, e.g. "
            "`conda run -n asrf python -m pip install mediapipe`."
        ) from exc
    return mp


def annotation_dir(raw_dir: str) -> str:
    root = os.path.join(raw_dir, "videos")
    for dirpath, _, filenames in os.walk(root):
        if "Video_TrainList.txt" in filenames and "Video_TestList.txt" in filenames:
            return dirpath
    raise FileNotFoundError("Cannot find IPN Hand annotations under {}".format(root))


def read_video_lengths(ann_dir: str) -> Dict[str, int]:
    lengths = {}
    for filename in ["Video_TrainList.txt", "Video_TestList.txt"]:
        with open(os.path.join(ann_dir, filename), "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video, frames = line.split()[:2]
                lengths[video] = int(frames)
    return lengths


def find_video_paths(raw_dir: str) -> Dict[str, str]:
    video_root = os.path.join(raw_dir, "videos")
    paths = {}
    for dirpath, _, filenames in os.walk(video_root):
        for filename in filenames:
            if filename.lower().endswith((".avi", ".mp4")):
                name = os.path.splitext(filename)[0]
                paths[name] = os.path.join(dirpath, filename)
    return paths


def select_videos(
    lengths: Dict[str, int],
    names: Optional[Iterable[str]],
    limit: Optional[int],
) -> List[str]:
    if names:
        selected = list(names)
    else:
        selected = sorted(lengths)
    if limit is not None:
        selected = selected[:limit]
    return selected


def hand_to_vector(hand_landmarks) -> np.ndarray:
    values = []
    for landmark in hand_landmarks.landmark:
        values.extend([landmark.x, landmark.y, landmark.z])
    return np.array(values, dtype=np.float32)


def frame_to_feature(results, feature_dim: int) -> Tuple[np.ndarray, float]:
    if not results.multi_hand_landmarks:
        return np.zeros(feature_dim, dtype=np.float32), 0.0

    hands = [hand_to_vector(hand) for hand in results.multi_hand_landmarks]
    handedness_scores = [
        float(item.classification[0].score)
        for item in results.multi_handedness or []
    ]

    hands = hands[:2]
    handedness_scores = handedness_scores[:2]
    while len(hands) < 2:
        hands.append(np.zeros(63, dtype=np.float32))
    while len(handedness_scores) < 2:
        handedness_scores.append(0.0)

    feature = np.concatenate(
        [hands[0], hands[1], np.array(handedness_scores, dtype=np.float32)]
    )
    return feature.astype(np.float32), max(handedness_scores)


def add_delta(features: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(features)
    delta[:, 1:] = features[:, 1:] - features[:, :-1]
    return np.concatenate([features, delta], axis=0)


def extract_one_video(
    video_path: str,
    n_frames: int,
    hands_model,
    output_path: str,
    delta_output_path: Optional[str],
) -> Tuple[int, float]:
    feature_dim = 63 * 2 + 2
    features = np.zeros((feature_dim, n_frames), dtype=np.float32)
    confidences = np.zeros(n_frames, dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video {}".format(video_path))

    last_feature = np.zeros(feature_dim, dtype=np.float32)
    read_count = 0
    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            feature = last_feature
            confidence = 0.0
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_model.process(rgb)
            feature, confidence = frame_to_feature(results, feature_dim)
            if confidence > 0:
                last_feature = feature
            read_count += 1

        features[:, idx] = feature
        confidences[idx] = confidence

    cap.release()
    np.save(output_path, features)
    np.save(os.path.splitext(output_path)[0] + "_confidence.npy", confidences)
    if delta_output_path is not None:
        np.save(delta_output_path, add_delta(features))
    return read_count, float(np.mean(confidences > 0))


def write_progress(path: str, rows: List[Dict[str, object]], append: bool = False) -> None:
    exists = os.path.exists(path)
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frames_expected",
                "frames_read",
                "detection_ratio",
                "feature_path",
            ],
        )
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_arguments()
    mp = import_mediapipe()
    ann_dir = annotation_dir(args.raw_dir)
    lengths = read_video_lengths(ann_dir)
    video_paths = find_video_paths(args.raw_dir)
    selected = select_videos(lengths, args.names, args.limit)

    output_dir = os.path.join(args.prepared_dir, args.output_name)
    delta_dir = os.path.join(args.prepared_dir, "features_keypoint_delta")
    progress_path = os.path.join(args.prepared_dir, "keypoint_extraction_log.csv")
    os.makedirs(output_dir, exist_ok=True)
    if args.with_delta:
        os.makedirs(delta_dir, exist_ok=True)

    progress_rows = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands_model:
        for idx, video in enumerate(selected, start=1):
            if video not in video_paths:
                print("skip {}: missing video file".format(video))
                continue

            output_path = os.path.join(output_dir, video + ".npy")
            delta_output_path = (
                os.path.join(delta_dir, video + ".npy") if args.with_delta else None
            )
            if (
                not args.overwrite
                and os.path.exists(output_path)
                and (delta_output_path is None or os.path.exists(delta_output_path))
            ):
                print("[{}/{}] skip existing {}".format(idx, len(selected), video))
                continue

            print("[{}/{}] extracting {}".format(idx, len(selected), video), flush=True)
            frames_read, detection_ratio = extract_one_video(
                video_paths[video],
                lengths[video],
                hands_model,
                output_path,
                delta_output_path,
            )
            progress_rows.append(
                {
                    "video": video,
                    "frames_expected": lengths[video],
                    "frames_read": frames_read,
                    "detection_ratio": detection_ratio,
                    "feature_path": output_path,
                }
            )
            write_progress(progress_path, [progress_rows[-1]], append=True)

    print("processed:", len(progress_rows))
    print("saved:", output_dir)
    if args.with_delta:
        print("saved:", delta_dir)


if __name__ == "__main__":
    main()
