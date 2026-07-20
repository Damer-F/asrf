import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


NON_GESTURE_LABELS = {"D0X", "B0A", "B0B"}


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare IPN Hand frame labels, boundary labels, event tables, and ASRF-style CSV files."
    )
    parser.add_argument(
        "--raw_dir",
        default="dataset/IPN Hand",
        help="IPN Hand root directory containing videos/ and annotations/",
    )
    parser.add_argument(
        "--output_dataset",
        default="dataset/ipn_hand",
        help="output dataset directory",
    )
    parser.add_argument(
        "--csv_output",
        default="csv/ipn_hand",
        help="output CSV directory",
    )
    parser.add_argument(
        "--feature_dir_name",
        default="features_keypoint",
        help="feature directory name that CSV files should point to",
    )
    parser.add_argument(
        "--extra_feature_dir_name",
        default="features_keypoint_delta",
        help="also create CSV files for this feature directory",
    )
    parser.add_argument(
        "--extra_csv_output",
        default="csv_delta/ipn_hand",
        help="CSV directory for the extra feature set",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=10,
        help="take every N-th official training video as validation",
    )
    return parser.parse_args()


def annotation_dir(raw_dir: str) -> str:
    root = os.path.join(raw_dir, "videos")
    for dirpath, _, filenames in os.walk(root):
        if "Annot_List.txt" in filenames and "Video_TrainList.txt" in filenames:
            return dirpath
    raise FileNotFoundError("Cannot find IPN Hand annotations under {}".format(root))


def read_video_list(path: str) -> Dict[str, int]:
    videos = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            videos[parts[0]] = int(parts[1])
    return videos


def read_annotations(path: str) -> Dict[str, List[Tuple[str, int, int]]]:
    rows = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("video,"):
                continue
            video, label, _class_id, start, end, _frames = line.split(",")[:6]
            rows[video].append((label, int(start), int(end)))
    return rows


def build_mapping(annotation_paths: Iterable[str]) -> Dict[str, int]:
    gestures = set()
    for path in annotation_paths:
        rows = read_annotations(path)
        for events in rows.values():
            for label, _start, _end in events:
                if label not in NON_GESTURE_LABELS:
                    gestures.add(label)

    labels = ["non_gesture"] + sorted(gestures)
    return {label: idx for idx, label in enumerate(labels)}


def label_to_id(label: str, class2id: Dict[str, int]) -> int:
    if label in NON_GESTURE_LABELS:
        return class2id["non_gesture"]
    return class2id[label]


def make_frame_arrays(
    video_name: str,
    n_frames: int,
    annotations: Dict[str, List[Tuple[str, int, int]]],
    class2id: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    labels = np.full(n_frames, class2id["non_gesture"], dtype=np.int64)
    event_rows = []

    for label, start, end in annotations.get(video_name, []):
        class_id = label_to_id(label, class2id)
        left = max(1, start) - 1
        right = min(n_frames, end)
        labels[left:right] = class_id
        event_rows.append(
            {
                "video": video_name,
                "raw_label": label,
                "class_name": "non_gesture" if label in NON_GESTURE_LABELS else label,
                "class_id": class_id,
                "start_frame": left,
                "end_frame": right - 1,
                "duration": right - left,
                "is_gesture": int(label not in NON_GESTURE_LABELS),
            }
        )

    boundary = np.zeros(n_frames, dtype=np.float32)
    boundary[0] = 1.0
    boundary[1:] = (labels[1:] != labels[:-1]).astype(np.float32)
    return labels, boundary, event_rows


def write_mapping(path: str, class2id: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for label, idx in sorted(class2id.items(), key=lambda x: x[1]):
            f.write("{} {}\n".format(idx, label))


def write_rows(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_train_val(train_videos: List[str], val_every: int) -> Tuple[List[str], List[str]]:
    train, val = [], []
    for idx, video in enumerate(train_videos):
        if val_every > 0 and idx % val_every == val_every - 1:
            val.append(video)
        else:
            train.append(video)
    return train, val


def make_csv_rows(
    videos: Iterable[str],
    output_dataset: str,
    feature_dir_name: str,
) -> List[Dict[str, str]]:
    rows = []
    for video in videos:
        rows.append(
            {
                "feature": os.path.join(output_dataset, feature_dir_name, video + ".npy"),
                "label": os.path.join(output_dataset, "gt_arr", video + ".npy"),
                "boundary": os.path.join(output_dataset, "gt_boundary_arr", video + ".npy"),
            }
        )
    return rows


def main() -> None:
    args = get_arguments()
    ann_dir = annotation_dir(args.raw_dir)

    train_videos = read_video_list(os.path.join(ann_dir, "Video_TrainList.txt"))
    test_videos = read_video_list(os.path.join(ann_dir, "Video_TestList.txt"))
    annotations = read_annotations(os.path.join(ann_dir, "Annot_List.txt"))
    class2id = build_mapping(
        [
            os.path.join(ann_dir, "Annot_TrainList.txt"),
            os.path.join(ann_dir, "Annot_TestList.txt"),
        ]
    )

    os.makedirs(args.output_dataset, exist_ok=True)
    gt_dir = os.path.join(args.output_dataset, "gt_arr")
    boundary_dir = os.path.join(args.output_dataset, "gt_boundary_arr")
    events_dir = os.path.join(args.output_dataset, "events")
    feature_dir = os.path.join(args.output_dataset, args.feature_dir_name)
    extra_feature_dir = os.path.join(args.output_dataset, args.extra_feature_dir_name)
    for path in [
        gt_dir,
        boundary_dir,
        events_dir,
        feature_dir,
        extra_feature_dir,
        args.csv_output,
    ]:
        os.makedirs(path, exist_ok=True)

    write_mapping(os.path.join(args.output_dataset, "mapping.txt"), class2id)

    all_videos = {**train_videos, **test_videos}
    event_rows = []
    summary_rows = []
    for video, n_frames in sorted(all_videos.items()):
        labels, boundary, video_events = make_frame_arrays(
            video, n_frames, annotations, class2id
        )
        np.save(os.path.join(gt_dir, video + ".npy"), labels)
        np.save(os.path.join(boundary_dir, video + ".npy"), boundary)
        event_rows.extend(video_events)
        summary_rows.append(
            {
                "video": video,
                "split": "train" if video in train_videos else "test",
                "frames": n_frames,
                "segments": len(video_events),
                "gesture_segments": sum(int(row["is_gesture"]) for row in video_events),
                "non_gesture_segments": sum(
                    1 - int(row["is_gesture"]) for row in video_events
                ),
            }
        )

    write_rows(
        os.path.join(events_dir, "events.csv"),
        event_rows,
        [
            "video",
            "raw_label",
            "class_name",
            "class_id",
            "start_frame",
            "end_frame",
            "duration",
            "is_gesture",
        ],
    )
    write_rows(
        os.path.join(events_dir, "video_summary.csv"),
        summary_rows,
        ["video", "split", "frames", "segments", "gesture_segments", "non_gesture_segments"],
    )

    train_names = sorted(train_videos)
    test_names = sorted(test_videos)
    train_names, val_names = split_train_val(train_names, args.val_every)

    for name, videos in [
        ("train1.csv", train_names),
        ("val1.csv", val_names),
        ("test1.csv", test_names),
    ]:
        rows = make_csv_rows(videos, args.output_dataset, args.feature_dir_name)
        write_rows(
            os.path.join(args.csv_output, name),
            rows,
            ["feature", "label", "boundary"],
        )

    extra_csv_output = args.extra_csv_output
    os.makedirs(extra_csv_output, exist_ok=True)
    for name, videos in [
        ("train1.csv", train_names),
        ("val1.csv", val_names),
        ("test1.csv", test_names),
    ]:
        rows = make_csv_rows(videos, args.output_dataset, args.extra_feature_dir_name)
        write_rows(
            os.path.join(extra_csv_output, name),
            rows,
            ["feature", "label", "boundary"],
        )

    print("annotation_dir:", ann_dir)
    print("classes:", len(class2id), class2id)
    print("videos:", len(all_videos))
    print("train:", len(train_names), "val:", len(val_names), "test:", len(test_names))
    print("events:", len(event_rows))
    print("saved:", args.output_dataset)
    print("saved:", args.csv_output)
    print("saved:", extra_csv_output)


if __name__ == "__main__":
    main()
