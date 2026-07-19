import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal action segment length distributions."
    )
    parser.add_argument("--dataset", required=True, choices=["gtea", "50salads", "breakfast"])
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--csv_dir", default="./csv")
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["all", "training", "validation", "test", "trainval"],
        default="all",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50, 100],
        help="segment length thresholds in frames",
    )
    parser.add_argument("--output", default=None, help="optional csv output path")
    return parser.parse_args()


def load_id2class(dataset: str, dataset_dir: str) -> Dict[int, str]:
    mapping_path = os.path.join(dataset_dir, dataset, "mapping.txt")
    id2class = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id2class[int(parts[0])] = parts[1]
    return id2class


def csv_label_paths(dataset: str, csv_dir: str, split: int, mode: str) -> List[str]:
    if mode == "trainval":
        modes = ["train", "val"]
    elif mode == "training":
        modes = ["train"]
    elif mode == "validation":
        modes = ["val"]
    else:
        modes = [mode]

    paths = []
    for prefix in modes:
        csv_path = os.path.join(csv_dir, dataset, "{}{}.csv".format(prefix, split))
        df = pd.read_csv(csv_path)
        paths.extend(df["label"].tolist())
    return paths


def all_label_paths(dataset: str, dataset_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dataset_dir, dataset, "gt_arr", "*.npy")))


def segments_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    if len(labels) == 0:
        return []

    segments = []
    start = 0
    current = int(labels[0])
    for i in range(1, len(labels)):
        label = int(labels[i])
        if label != current:
            segments.append((current, i - start))
            start = i
            current = label
    segments.append((current, len(labels) - start))
    return segments


def summarize_segments(
    dataset: str,
    label_paths: List[str],
    id2class: Dict[int, str],
    thresholds: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in label_paths:
        labels = np.load(path).astype(np.int64)
        video = os.path.basename(path)[:-4]
        for class_id, length in segments_from_labels(labels):
            rows.append(
                {
                    "dataset": dataset,
                    "video": video,
                    "class_id": class_id,
                    "class_name": id2class.get(class_id, "class_{}".format(class_id)),
                    "length": length,
                }
            )

    df = pd.DataFrame(rows)
    overall = {
        "dataset": dataset,
        "videos": len(label_paths),
        "segments": len(df),
        "mean_length": df["length"].mean(),
        "median_length": df["length"].median(),
        "min_length": df["length"].min(),
        "max_length": df["length"].max(),
    }
    for threshold in thresholds:
        overall["segments_le_{}".format(threshold)] = int((df["length"] <= threshold).sum())
        overall["ratio_le_{}".format(threshold)] = float((df["length"] <= threshold).mean())

    per_class = (
        df.groupby(["class_id", "class_name"])["length"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    for threshold in thresholds:
        short_counts = (
            df[df["length"] <= threshold]
            .groupby(["class_id", "class_name"])["length"]
            .count()
            .rename("segments_le_{}".format(threshold))
            .reset_index()
        )
        per_class = per_class.merge(
            short_counts, on=["class_id", "class_name"], how="left"
        )
        col = "segments_le_{}".format(threshold)
        per_class[col] = per_class[col].fillna(0).astype(int)
        per_class["ratio_le_{}".format(threshold)] = per_class[col] / per_class["count"]

    return pd.DataFrame([overall]), per_class


def main() -> None:
    args = get_arguments()
    id2class = load_id2class(args.dataset, args.dataset_dir)

    if args.mode == "all":
        label_paths = all_label_paths(args.dataset, args.dataset_dir)
    else:
        if args.split is None:
            raise ValueError("--split is required when --mode is not all")
        label_paths = csv_label_paths(args.dataset, args.csv_dir, args.split, args.mode)

    overall, per_class = summarize_segments(
        args.dataset, label_paths, id2class, args.thresholds
    )

    print("Overall")
    print(overall.to_string(index=False))
    print("\nPer-class short segment summary")
    print(per_class.to_string(index=False))

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        overall_path = args.output
        class_path = os.path.splitext(args.output)[0] + "_per_class.csv"
        overall.to_csv(overall_path, index=False)
        per_class.to_csv(class_path, index=False)
        print("\nSaved:", overall_path)
        print("Saved:", class_path)


if __name__ == "__main__":
    main()
