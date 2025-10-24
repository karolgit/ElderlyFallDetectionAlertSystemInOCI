from __future__ import annotations

from typing import List, Tuple, Optional
import cv2

from .schemas import PersonPose

# Skeleton connections using COCO keypoint names
SKELETON_PAIRS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _kp_by_name(person: PersonPose) -> dict[str, Tuple[float, float, float]]:
    m: dict[str, Tuple[float, float, float]] = {}
    for kp in person.keypoints:
        if kp.name:
            m[kp.name] = (kp.x, kp.y, kp.score)
    return m


def draw_skeleton(frame_bgr, people: List[PersonPose], score_thresh: float = 0.3) -> None:
    h, w = frame_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * max(h, w))))
    radius = max(2, int(round(0.006 * max(h, w))))

    # Colors in BGR
    kp_color = (0, 255, 136)
    edge_color = (0, 255, 136)
    box_color = (0, 170, 255)

    for person in people:
        by_name = _kp_by_name(person)
        # Draw keypoints
        for name, (x, y, s) in by_name.items():
            if s < score_thresh:
                continue
            cv2.circle(frame_bgr, (int(round(x)), int(round(y))), radius, kp_color, -1, lineType=cv2.LINE_AA)
        # Draw skeleton
        for a, b in SKELETON_PAIRS:
            if a in by_name and b in by_name:
                x1, y1, s1 = by_name[a]
                x2, y2, s2 = by_name[b]
                if s1 >= score_thresh and s2 >= score_thresh:
                    cv2.line(
                        frame_bgr,
                        (int(round(x1)), int(round(y1))),
                        (int(round(x2)), int(round(y2))),
                        edge_color,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )
        # Draw bbox
        if person.bbox and len(person.bbox) == 4:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(
                frame_bgr,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                box_color,
                thickness,
                lineType=cv2.LINE_AA,
            )
