import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class Event:
    video: str
    class_id: int
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate event-level gesture detection and non-gesture false triggers."
    )
    parser.add_argument("array_dir", help="prediction_arrays directory")
    parser.add_argument("--dataset_dir", default="dataset/ipn_hand")
    parser.add_argument("--non_gesture_id", type=int, default=0)
    parser.add_argument("--pred_kind", choices=["pred", "refined_pred"], default="refined_pred")
    parser.add_argument("--iou_threshold", type=float, default=0.25)
    parser.add_argument("--min_duration", type=int, default=8)
    parser.add_argument("--merge_gap", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def find_names(array_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(array_dir, "*_gt.npy")))
    return [os.path.basename(path)[: -len("_gt.npy")] for path in paths]


def segments_from_labels(
    labels: np.ndarray,
    video: str,
    non_gesture_id: int,
) -> List[Event]:
    events = []
    if len(labels) == 0:
        return events
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            if current != non_gesture_id:
                events.append(Event(video, current, start, idx - 1))
            start = idx
            current = label
    if current != non_gesture_id:
        events.append(Event(video, current, start, len(labels) - 1))
    return events


def merge_events(events: Sequence[Event], merge_gap: int) -> List[Event]:
    if not events:
        return []
    merged = [events[0]]
    for event in events[1:]:
        last = merged[-1]
        if (
            event.video == last.video
            and event.class_id == last.class_id
            and event.start - last.end - 1 <= merge_gap
        ):
            merged[-1] = Event(last.video, last.class_id, last.start, event.end)
        else:
            merged.append(event)
    return merged


def filter_events(events: Sequence[Event], min_duration: int, merge_gap: int) -> List[Event]:
    events = [event for event in events if event.duration >= min_duration]
    events = merge_events(events, merge_gap)
    return [event for event in events if event.duration >= min_duration]


def event_iou(a: Event, b: Event) -> float:
    if a.video != b.video or a.class_id != b.class_id:
        return 0.0
    inter = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    union = max(a.end, b.end) - min(a.start, b.start) + 1
    return inter / union if union > 0 else 0.0


def match_events(
    pred_events: Sequence[Event],
    gt_events: Sequence[Event],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    matched_gt = set()
    tp = 0
    for pred_idx, pred in enumerate(pred_events):
        best_iou = 0.0
        best_gt_idx = None
        for gt_idx, gt in enumerate(gt_events):
            if gt_idx in matched_gt:
                continue
            iou = event_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            tp += 1

    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    return tp, fp, fn


def count_non_gesture_false_triggers(
    pred_events: Sequence[Event],
    gt_labels_by_video: Dict[str, np.ndarray],
    non_gesture_id: int,
) -> int:
    count = 0
    for event in pred_events:
        gt = gt_labels_by_video[event.video]
        start = max(0, event.start)
        end = min(len(gt) - 1, event.end)
        if start > end:
            continue
        if np.all(gt[start : end + 1] == non_gesture_id):
            count += 1
    return count


def load_id2class(dataset_dir: str) -> Dict[int, str]:
    id2class = {}
    with open(os.path.join(dataset_dir, "mapping.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id2class[int(parts[0])] = parts[1]
    return id2class


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_arguments()
    id2class = load_id2class(args.dataset_dir)
    names = find_names(args.array_dir)

    all_gt_events = []
    all_pred_events = []
    gt_labels_by_video = {}
    total_frames = 0

    per_video_rows = []
    for name in names:
        gt = np.load(os.path.join(args.array_dir, name + "_gt.npy")).astype(np.int64)
        pred = np.load(os.path.join(args.array_dir, name + "_" + args.pred_kind + ".npy")).astype(np.int64)
        length = min(len(gt), len(pred))
        gt = gt[:length]
        pred = pred[:length]
        gt_labels_by_video[name] = gt
        total_frames += length

        gt_events = filter_events(
            segments_from_labels(gt, name, args.non_gesture_id),
            args.min_duration,
            args.merge_gap,
        )
        pred_events = filter_events(
            segments_from_labels(pred, name, args.non_gesture_id),
            args.min_duration,
            args.merge_gap,
        )
        tp, fp, fn = match_events(pred_events, gt_events, args.iou_threshold)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        false_triggers = count_non_gesture_false_triggers(
            pred_events, {name: gt}, args.non_gesture_id
        )
        per_video_rows.append(
            {
                "video": name,
                "gt_events": len(gt_events),
                "pred_events": len(pred_events),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "non_gesture_false_triggers": false_triggers,
            }
        )

        all_gt_events.extend(gt_events)
        all_pred_events.extend(pred_events)

    tp, fp, fn = match_events(all_pred_events, all_gt_events, args.iou_threshold)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_triggers = count_non_gesture_false_triggers(
        all_pred_events, gt_labels_by_video, args.non_gesture_id
    )
    minutes = total_frames / args.fps / 60.0
    false_triggers_per_min = safe_div(false_triggers, minutes)

    summary = {
        "pred_kind": args.pred_kind,
        "iou_threshold": args.iou_threshold,
        "min_duration": args.min_duration,
        "merge_gap": args.merge_gap,
        "videos": len(names),
        "gt_events": len(all_gt_events),
        "pred_events": len(all_pred_events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "non_gesture_false_triggers": false_triggers,
        "minutes": minutes,
        "false_triggers_per_min": false_triggers_per_min,
    }

    print("Event summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print("{}: {:.4f}".format(key, value))
        else:
            print("{}: {}".format(key, value))

    if args.output is not None:
        base, ext = os.path.splitext(args.output)
        write_csv(args.output, [summary])
        write_csv(base + "_per_video" + ext, per_video_rows)
        print("saved:", args.output)
        print("saved:", base + "_per_video" + ext)


if __name__ == "__main__":
    main()
