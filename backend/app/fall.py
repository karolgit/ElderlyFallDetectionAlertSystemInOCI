from __future__ import annotations

from typing import List
import math

from .schemas import PersonPose


class FallDetector:
    def __init__(self, angle_threshold_deg: float = 50.0) -> None:
        self.angle_threshold_deg = angle_threshold_deg

    def _get_point(self, person: PersonPose, name: str):
        for kp in person.keypoints:
            if kp.name == name:
                return kp
        return None

    def _center(self, a, b):
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    def _torso_angle_deg(self, person: PersonPose) -> float | None:
        left_shoulder = self._get_point(person, "left_shoulder")
        right_shoulder = self._get_point(person, "right_shoulder")
        left_hip = self._get_point(person, "left_hip")
        right_hip = self._get_point(person, "right_hip")
        if not (left_shoulder and right_shoulder and left_hip and right_hip):
            return None
        sx, sy = self._center(left_shoulder, right_shoulder)
        hx, hy = self._center(left_hip, right_hip)
        dx, dy = sx - hx, sy - hy
        # Angle relative to vertical: 0 = upright, 90 = horizontal
        angle_rad = abs(math.atan2(dx, dy))
        return math.degrees(angle_rad)

    def _aspect_ratio(self, person: PersonPose) -> float | None:
        if not person.bbox:
            return None
        x1, y1, x2, y2 = person.bbox
        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        return float(w / h)

    def score_person(self, person: PersonPose) -> float:
        angle = self._torso_angle_deg(person)
        ar = self._aspect_ratio(person)
        score = 0.0
        # Angle component: map threshold to ~0.6
        if angle is not None:
            score_angle = min(1.0, max(0.0, (angle - 20.0) / (90.0 - 20.0)))
            score = max(score, score_angle)
        # Aspect ratio component: horizontal (w>h) suggests fall
        if ar is not None:
            score_ar = min(1.0, max(0.0, (ar - 0.8) / (2.0 - 0.8)))
            score = max(score, score_ar)
        # Combine with person detection confidence
        score = score * max(0.5, min(1.0, person.score))
        return float(score)

    def predict(self, people: List[PersonPose]) -> tuple[bool, float]:
        if not people:
            return False, 0.0
        scores = [self.score_person(p) for p in people]
        fall_score = float(max(scores))
        is_fall = fall_score >= 0.6
        return is_fall, fall_score
