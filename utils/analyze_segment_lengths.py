import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal action segment length distributions."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        choices=["gtea", "50salads", "breakfast"],
        help="one or more datasets to analyze",
    )
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
        default=[5, 10, 25, 50, 100, 200, 500, 1000],
        help="segment length thresholds in frames",
    )
    parser.add_argument("--output", default=None, help="optional csv output path")
    parser.add_argument(
        "--plot_heatmap",
        action="store_true",
        help="save heatmaps for segment length distribution",
    )
    parser.add_argument(
        "--heatmap_dir",
        default="export/segment_stats",
        help="directory where heatmap images will be saved",
    )
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


def build_segment_dataframe(
    dataset: str,
    label_paths: List[str],
    id2class: Dict[int, str],
) -> pd.DataFrame:
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

    return pd.DataFrame(rows)


def summarize_segments(
    dataset: str,
    label_paths: List[str],
    id2class: Dict[int, str],
    thresholds: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = build_segment_dataframe(dataset, label_paths, id2class)
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

    return pd.DataFrame([overall]), per_class, df


def bin_labels(edges: List[int]) -> List[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            labels.append(">{}".format(left))
        else:
            labels.append("{}-{}".format(left + 1, right))
    return labels


def add_length_bins(df: pd.DataFrame, edges: List[int]) -> pd.DataFrame:
    labels = bin_labels(edges)
    out = df.copy()
    out["length_bin"] = pd.cut(
        out["length"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return out


def plot_overall_heatmap(dfs: List[pd.DataFrame], edges: List[int], output_dir: str) -> str:
    data = []
    labels = bin_labels(edges)
    for df in dfs:
        dataset = df["dataset"].iloc[0]
        binned = add_length_bins(df, edges)
        counts = binned["length_bin"].value_counts().reindex(labels, fill_value=0)
        ratios = counts / counts.sum()
        data.append({"dataset": dataset, **ratios.to_dict()})

    matrix = pd.DataFrame(data).set_index("dataset")[labels]
    fig, ax = plt.subplots(figsize=(12, max(2.0, 0.7 * len(matrix))))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Segment length bin (frames)")
    ax.set_title("Overall action segment length distribution")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if value > 0:
                ax.text(j, i, "{:.1f}%".format(value * 100), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Ratio of all segments")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "segment_length_overall_heatmap.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_heatmaps(dfs: List[pd.DataFrame], edges: List[int], output_dir: str) -> str:
    labels = bin_labels(edges)
    n_datasets = len(dfs)
    fig_height = sum(max(3.0, 0.35 * df["class_name"].nunique()) for df in dfs)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(15, fig_height), squeeze=False)

    for ax, df in zip(axes[:, 0], dfs):
        dataset = df["dataset"].iloc[0]
        binned = add_length_bins(df, edges)
        counts = pd.crosstab(binned["class_name"], binned["length_bin"]).reindex(
            columns=labels, fill_value=0
        )
        ratios = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        ratios = ratios.loc[counts.sum(axis=1).sort_values(ascending=False).index]

        im = ax.imshow(ratios.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_title("{} per-class segment length distribution".format(dataset))
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ratios.index)), labels=ratios.index)
        ax.set_xlabel("Segment length bin (frames)")

        for i in range(ratios.shape[0]):
            for j in range(ratios.shape[1]):
                value = ratios.iat[i, j]
                if value >= 0.15:
                    ax.text(j, i, "{:.0f}%".format(value * 100), ha="center", va="center", fontsize=7)

    fig.subplots_adjust(left=0.18, right=0.86, hspace=0.42, bottom=0.08, top=0.95)
    cbar_ax = fig.add_axes([0.89, 0.16, 0.02, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="Ratio within class")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "segment_length_class_heatmap.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = get_arguments()
    all_overall = []
    all_per_class = []
    segment_dfs = []

    for dataset in args.dataset:
        id2class = load_id2class(dataset, args.dataset_dir)

        if args.mode == "all":
            label_paths = all_label_paths(dataset, args.dataset_dir)
        else:
            if args.split is None:
                raise ValueError("--split is required when --mode is not all")
            label_paths = csv_label_paths(dataset, args.csv_dir, args.split, args.mode)

        overall, per_class, segments = summarize_segments(
            dataset, label_paths, id2class, args.thresholds
        )
        all_overall.append(overall)
        all_per_class.append(per_class)
        segment_dfs.append(segments)

        print("\n== {} ==".format(dataset))
        print("Overall")
        print(overall.to_string(index=False))
        print("\nPer-class short segment summary")
        print(per_class.to_string(index=False))

        if args.output is not None and len(args.dataset) == 1:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            overall_path = args.output
            class_path = os.path.splitext(args.output)[0] + "_per_class.csv"
            overall.to_csv(overall_path, index=False)
            per_class.to_csv(class_path, index=False)
            print("\nSaved:", overall_path)
            print("Saved:", class_path)

    if args.output is not None and len(args.dataset) > 1:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        overall_path = args.output
        class_path = os.path.splitext(args.output)[0] + "_per_class.csv"
        pd.concat(all_overall, ignore_index=True).to_csv(overall_path, index=False)
        pd.concat(all_per_class, ignore_index=True).to_csv(class_path, index=False)
        print("\nSaved:", overall_path)
        print("Saved:", class_path)

    if args.plot_heatmap:
        edges = [0] + args.thresholds + [10 ** 9]
        overall_path = plot_overall_heatmap(segment_dfs, edges, args.heatmap_dir)
        class_path = plot_class_heatmaps(segment_dfs, edges, args.heatmap_dir)
        print("\nSaved:", overall_path)
        print("Saved:", class_path)


if __name__ == "__main__":
    main()
