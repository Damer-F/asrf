import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GTEA raw videos with aligned GT/pred/refined/boundary timelines."
    )
    parser.add_argument(
        "array_dir",
        type=str,
        help="prediction array directory, e.g. result_paperlike/gtea_split1_50ep/prediction_arrays",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="dataset/gtea_raw/gtea_videos/Videos",
        help="directory containing raw GTEA mp4 videos",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="dataset root containing gtea/mapping.txt",
    )
    parser.add_argument(
        "--raw_label_dir",
        type=str,
        default="dataset/gtea_raw/gtea_labels_71/labels",
        help="directory containing raw GTEA 71-class label txt files",
    )
    parser.add_argument(
        "--gt_source",
        choices=["raw_label", "prediction_array"],
        default="raw_label",
        help="source for the GT row. raw_label converts raw GTEA labels to verb-level frame labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory. Defaults to sibling video_timelines/ for prediction_arrays input.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="optional video names without extension",
    )
    parser.add_argument("--boundary_th", type=float, default=0.5)
    parser.add_argument("--max_width", type=int, default=960)
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="fourcc codec for output mp4",
    )
    return parser.parse_args()


def load_id2class(dataset_dir: str) -> Dict[int, str]:
    mapping_path = os.path.join(dataset_dir, "gtea", "mapping.txt")
    id2class = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id2class[int(parts[0])] = parts[1]
    return id2class


def load_class2id(dataset_dir: str) -> Dict[str, int]:
    return {v: k for k, v in load_id2class(dataset_dir).items()}


def find_names(array_dir: str, names: Optional[Sequence[str]]) -> List[str]:
    if names:
        return list(names)
    paths = sorted(glob.glob(os.path.join(array_dir, "*_gt.npy")))
    return [os.path.basename(path)[: -len("_gt.npy")] for path in paths]


def default_output_dir(array_dir: str) -> str:
    normalized = os.path.normpath(array_dir)
    parent = os.path.dirname(normalized)
    if os.path.basename(normalized) == "prediction_arrays":
        return os.path.join(parent, "video_timelines")
    return os.path.join(normalized, "video_timelines")


def make_palette(n_classes: int) -> np.ndarray:
    base = np.array(
        [
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
            [23, 190, 207],
            [210, 210, 210],
            [57, 59, 121],
            [82, 84, 163],
            [107, 110, 207],
            [156, 158, 222],
            [99, 121, 57],
            [140, 162, 82],
            [181, 207, 107],
            [206, 219, 156],
            [140, 109, 49],
        ],
        dtype=np.uint8,
    )
    if n_classes <= len(base):
        rgb = base[:n_classes]
    else:
        extra = np.random.default_rng(0).integers(
            40, 230, size=(n_classes - len(base), 3), dtype=np.uint8
        )
        rgb = np.vstack([base, extra])
    return rgb[:, ::-1].copy()


def raw_label_to_verb_array(
    raw_label_path: str,
    length: int,
    class2id: Dict[str, int],
) -> np.ndarray:
    labels = np.full(length, class2id["background"], dtype=np.int64)
    pattern = re.compile(r"<([^>]+)>.*\((\d+)-(\d+)\) \[[01]\]")

    with open(raw_label_path, "r") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            verb = match.group(1)
            if verb not in class2id:
                continue
            start = int(match.group(2))
            end = int(match.group(3))
            left = max(1, start) - 1
            right = min(length, end)
            labels[left:right] = class2id[verb]

    return labels


def load_arrays(array_dir: str, name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    required = [
        os.path.join(array_dir, name + "_gt.npy"),
        os.path.join(array_dir, name + "_pred.npy"),
        os.path.join(array_dir, name + "_refined_pred.npy"),
        os.path.join(array_dir, name + "_boundary.npy"),
    ]
    if not all(os.path.exists(path) for path in required):
        return None

    gt = np.load(required[0]).astype(np.int64)
    gt = gt[gt != 255]
    pred = np.load(required[1]).astype(np.int64)[: len(gt)]
    refined = np.load(required[2]).astype(np.int64)[: len(gt)]
    boundary = np.load(required[3]).astype(np.float32)[: len(gt)]
    return gt, pred, refined, boundary


def timeline_bar(
    labels: np.ndarray,
    palette: np.ndarray,
    width: int,
    height: int,
    cursor: int,
) -> np.ndarray:
    idx = np.linspace(0, len(labels) - 1, width).astype(np.int64)
    colors = palette[labels[idx]]
    bar = np.repeat(colors[np.newaxis, :, :], height, axis=0)
    x = int(round(cursor / max(1, len(labels) - 1) * (width - 1)))
    cv2.line(bar, (x, 0), (x, height - 1), (0, 0, 0), 2)
    cv2.line(bar, (x, 0), (x, height - 1), (255, 255, 255), 1)
    return bar


def boundary_bar(
    boundary: np.ndarray,
    width: int,
    height: int,
    cursor: int,
    threshold: float,
) -> np.ndarray:
    idx = np.linspace(0, len(boundary) - 1, width).astype(np.int64)
    values = np.clip(boundary[idx], 0.0, 1.0)
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    red = (values * 255).astype(np.uint8)
    bar[:, :, 2] = red
    bar[:, :, 1] = (255 - 0.55 * red).astype(np.uint8)
    hits = np.where(values >= threshold)[0]
    bar[:, hits, :] = np.array([30, 30, 30], dtype=np.uint8)
    x = int(round(cursor / max(1, len(boundary) - 1) * (width - 1)))
    cv2.line(bar, (x, 0), (x, height - 1), (0, 0, 0), 2)
    cv2.line(bar, (x, 0), (x, height - 1), (255, 255, 255), 1)
    return bar


def draw_label(
    canvas: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.5,
    color: Tuple[int, int, int] = (35, 35, 35),
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def make_panel(
    width: int,
    cursor: int,
    gt: np.ndarray,
    pred: np.ndarray,
    refined: np.ndarray,
    boundary: np.ndarray,
    palette: np.ndarray,
    id2class: Dict[int, str],
    boundary_th: float,
) -> np.ndarray:
    label_w = 126
    row_h = 20
    gap = 8
    header_h = 44
    panel_h = header_h + 4 * row_h + 5 * gap + 8
    panel = np.full((panel_h, width, 3), 250, dtype=np.uint8)

    cursor = min(cursor, len(gt) - 1)
    gt_name = id2class.get(int(gt[cursor]), str(int(gt[cursor])))
    pred_name = id2class.get(int(pred[cursor]), str(int(pred[cursor])))
    ref_name = id2class.get(int(refined[cursor]), str(int(refined[cursor])))
    title = "frame {} | GT: {} | Before: {} | Refine: {} | Boundary: {:.2f}".format(
        cursor, gt_name, pred_name, ref_name, float(boundary[cursor])
    )
    draw_label(panel, title, (10, 27), scale=0.55)

    rows = [
        ("GT", timeline_bar(gt, palette, width - label_w - 12, row_h, cursor)),
        (
            "Before refine",
            timeline_bar(pred, palette, width - label_w - 12, row_h, cursor),
        ),
        (
            "Refine",
            timeline_bar(refined, palette, width - label_w - 12, row_h, cursor),
        ),
        (
            "Boundary",
            boundary_bar(boundary, width - label_w - 12, row_h, cursor, boundary_th),
        ),
    ]

    y = header_h
    for label, bar in rows:
        draw_label(panel, label, (10, y + 15), scale=0.48)
        panel[y : y + row_h, label_w : label_w + bar.shape[1]] = bar
        y += row_h + gap

    return panel


def render_one(
    array_dir: str,
    video_dir: str,
    output_dir: str,
    name: str,
    id2class: Dict[int, str],
    class2id: Dict[str, int],
    raw_label_dir: str,
    gt_source: str,
    boundary_th: float,
    max_width: int,
    codec: str,
) -> Optional[str]:
    arrays = load_arrays(array_dir, name)
    if arrays is None:
        print("skip {}: missing prediction arrays".format(name))
        return None

    video_path = os.path.join(video_dir, name + ".mp4")
    if not os.path.exists(video_path):
        print("skip {}: missing raw video {}".format(name, video_path))
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("skip {}: cannot open video".format(name))
        return None

    gt_from_prediction, pred, refined, boundary = arrays
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(len(gt_from_prediction), len(pred), len(refined), len(boundary), video_frames)
    gt_from_prediction = gt_from_prediction[:total]
    pred = pred[:total]
    refined = refined[:total]
    boundary = boundary[:total]

    if gt_source == "raw_label":
        raw_label_path = os.path.join(raw_label_dir, name + ".txt")
        if not os.path.exists(raw_label_path):
            cap.release()
            print("skip {}: missing raw label {}".format(name, raw_label_path))
            return None
        gt = raw_label_to_verb_array(raw_label_path, total, class2id)
        mismatches = int(np.sum(gt != gt_from_prediction))
        if mismatches > 0:
            print(
                "warning {}: raw label GT differs from prediction-array GT at {} frame(s)".format(
                    name, mismatches
                )
            )
    else:
        gt = gt_from_prediction

    n_classes = int(max(gt.max(), pred.max(), refined.max())) + 1
    palette = make_palette(n_classes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / max(1, src_w))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))

    probe_panel = make_panel(
        out_w, 0, gt, pred, refined, boundary, palette, id2class, boundary_th
    )
    frame_h = out_h + probe_panel.shape[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, name + "_video_timeline.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (out_w, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        print("skip {}: cannot create output writer".format(name))
        return None

    written = 0
    for t in range(total):
        ok, frame = cap.read()
        if not ok:
            if written == 0:
                break
            frame = last_frame.copy()
        else:
            last_frame = frame

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        panel = make_panel(
            out_w, t, gt, pred, refined, boundary, palette, id2class, boundary_th
        )
        composed = np.vstack([frame, panel])
        writer.write(composed)
        written += 1

    cap.release()
    writer.release()
    print("saved {} ({} frames)".format(out_path, written))
    return out_path


def main() -> None:
    args = get_arguments()
    id2class = load_id2class(args.dataset_dir)
    class2id = load_class2id(args.dataset_dir)
    output_dir = args.output_dir or default_output_dir(args.array_dir)
    names = find_names(args.array_dir, args.names)

    saved = []
    for name in names:
        out_path = render_one(
            args.array_dir,
            args.video_dir,
            output_dir,
            name,
            id2class,
            class2id,
            args.raw_label_dir,
            args.gt_source,
            args.boundary_th,
            args.max_width,
            args.codec,
        )
        if out_path is not None:
            saved.append(out_path)
    print("saved {} video timeline(s)".format(len(saved)))


if __name__ == "__main__":
    main()
