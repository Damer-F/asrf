import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize extracted IPN Hand keypoints on raw video frames."
    )
    parser.add_argument("--raw_dir", default="dataset/IPN Hand")
    parser.add_argument("--feature_dir", default="dataset/ipn_hand/features_keypoint")
    parser.add_argument("--output_dir", default="export/exp_result/ipn_keypoint_preview")
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--video_seconds", type=float, default=8.0)
    parser.add_argument("--max_width", type=int, default=720)
    return parser.parse_args()


def find_video_paths(raw_dir: str) -> Dict[str, str]:
    video_root = os.path.join(raw_dir, "videos")
    paths = {}
    for dirpath, _, filenames in os.walk(video_root):
        for filename in filenames:
            if filename.lower().endswith((".avi", ".mp4")):
                paths[os.path.splitext(filename)[0]] = os.path.join(dirpath, filename)
    return paths


def hand_points(feature: np.ndarray, hand_idx: int, width: int, height: int) -> Optional[np.ndarray]:
    start = hand_idx * 63
    vals = feature[start : start + 63]
    if np.allclose(vals, 0):
        return None
    pts = vals.reshape(21, 3)
    xy = np.zeros((21, 2), dtype=np.int32)
    xy[:, 0] = np.clip(np.round(pts[:, 0] * width), 0, width - 1).astype(np.int32)
    xy[:, 1] = np.clip(np.round(pts[:, 1] * height), 0, height - 1).astype(np.int32)
    return xy


def draw_hand(frame: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
    for i, point in enumerate(points):
        radius = 4 if i in [0, 4, 8, 12, 16, 20] else 3
        cv2.circle(frame, tuple(point), radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), radius, color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, feature: np.ndarray, frame_idx: int, confidence: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    colors = [(0, 220, 255), (255, 90, 40)]
    for hand_idx, color in enumerate(colors):
        points = hand_points(feature, hand_idx, w, h)
        if points is not None:
            draw_hand(out, points, color)
    text = "frame {} | detection confidence {:.2f}".format(frame_idx, confidence)
    cv2.rectangle(out, (8, 8), (430, 38), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / max(1, w))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def make_contact_sheet(frames: List[np.ndarray], cols: int = 4) -> np.ndarray:
    if not frames:
        raise ValueError("no frames to render")
    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    return sheet


def visualize_one(
    name: str,
    video_path: str,
    feature_dir: str,
    output_dir: str,
    samples: int,
    video_seconds: float,
    max_width: int,
) -> None:
    feature_path = os.path.join(feature_dir, name + ".npy")
    confidence_path = os.path.join(feature_dir, name + "_confidence.npy")
    if not os.path.exists(feature_path):
        print("skip {}: missing feature".format(name))
        return
    features = np.load(feature_path).astype(np.float32)
    confidences = np.load(confidence_path).astype(np.float32) if os.path.exists(confidence_path) else np.zeros(features.shape[1], dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("skip {}: cannot open video".format(name))
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), features.shape[1])
    os.makedirs(output_dir, exist_ok=True)

    sample_indices = np.linspace(0, n_frames - 1, samples).astype(np.int64)
    preview_frames = []
    for idx in sample_indices:
        frame = read_frame(cap, int(idx))
        if frame is None:
            continue
        frame = resize_frame(frame, max_width)
        overlay = draw_overlay(frame, features[:, idx], int(idx), float(confidences[idx]))
        preview_frames.append(overlay)

    if preview_frames:
        sheet = make_contact_sheet(preview_frames)
        preview_path = os.path.join(output_dir, name + "_keypoints_preview.jpg")
        cv2.imwrite(preview_path, sheet)
        print("saved", preview_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    if not ok:
        cap.release()
        return
    first = resize_frame(first, max_width)
    out_h, out_w = first.shape[:2]
    out_path = os.path.join(output_dir, name + "_keypoints_overlay.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    max_frames = min(n_frames, int(round(video_seconds * fps)))
    for idx in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_frame(frame, max_width)
        writer.write(draw_overlay(frame, features[:, idx], idx, float(confidences[idx])))
    writer.release()
    cap.release()
    print("saved", out_path)


def main() -> None:
    args = get_arguments()
    video_paths = find_video_paths(args.raw_dir)
    for name in args.names:
        if name not in video_paths:
            print("skip {}: missing raw video".format(name))
            continue
        visualize_one(
            name,
            video_paths[name],
            args.feature_dir,
            args.output_dir,
            args.samples,
            args.video_seconds,
            args.max_width,
        )


if __name__ == "__main__":
    main()
