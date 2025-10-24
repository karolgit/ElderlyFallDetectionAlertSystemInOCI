from __future__ import annotations

from typing import List, Optional, Tuple

import logging
import math
import torch
from PIL import Image

from .device import get_torch_device
from .schemas import Keypoint, PersonPose
from .utils import COCO_KEYPOINT_NAMES

logger = logging.getLogger(__name__)


class PoseEstimator:
    def __init__(self, preferred_device: Optional[str] = None, score_threshold: float = 0.5, max_side: int = 640) -> None:
        self.device, self.device_type = get_torch_device(preferred_device)
        self.inference_device = self.device  # may be overridden for torchvision on MPS
        self.score_threshold = score_threshold
        self.max_side = max_side
        self.backend: Optional[str] = None  # "sports2d" | "torchvision"
        self.model = None
        self.weights = None

        logger.debug("Initializing PoseEstimator on device=%s", self.device)

        # Try Sports2D first
        try:
            import importlib
            sports2d_spec = importlib.util.find_spec("sports2d") or importlib.util.find_spec("Sports2D")
            if sports2d_spec is not None:
                self._init_sports2d()
                self.backend = "sports2d"
                logger.debug("Using Sports2D backend for pose estimation")
        except Exception as e:
            logger.debug("Sports2D not available or failed to init: %s", e)
            self.backend = None

        # Fallback to torchvision Keypoint R-CNN
        if self.backend is None:
            self._init_torchvision()
            self.backend = "torchvision"
            # Known issues with torchvision detection models on MPS -> run on CPU for stability
            if self.device.type == "mps":
                self.inference_device = torch.device("cpu")
                logger.debug("Routing torchvision inference to CPU due to MPS incompatibilities")
            logger.debug("Using torchvision KeypointRCNN backend for pose estimation")

        assert self.model is not None
        self.model.eval()

    def _init_sports2d(self) -> None:
        import sports2d  # type: ignore
        self.model = sports2d.load_model(device=self.device)

    def _init_torchvision(self) -> None:
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights)
        # Temporarily move to CPU; may change inference_device later
        model.to(torch.device("cpu"))
        self.model = model
        self.weights = weights
        logger.debug("Loaded KeypointRCNN weights: %s", getattr(weights, "__class__", type(weights)).__name__)

    @torch.inference_mode()
    def estimate(self, image: Image.Image) -> List[PersonPose]:
        if self.backend == "sports2d":
            return self._estimate_sports2d(image)
        return self._estimate_torchvision(image)

    def _estimate_sports2d(self, image: Image.Image) -> List[PersonPose]:
        outputs = self.model.predict(image)  # type: ignore[attr-defined]
        people: List[PersonPose] = []
        for pose in outputs:
            kps = [
                Keypoint(x=float(k[0]), y=float(k[1]), score=float(k[2]))
                for k in pose["keypoints"]
            ]
            people.append(PersonPose(keypoints=kps, score=float(pose.get("score", 1.0)), bbox=pose.get("bbox")))
        logger.debug("Sports2D detected %d person(s)", len(people))
        return people

    def _resize_for_inference(self, image: Image.Image) -> Tuple[Image.Image, float]:
        w, h = image.size
        scale = 1.0
        max_wh = max(w, h)
        if max_wh > self.max_side:
            scale = self.max_side / float(max_wh)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            logger.debug("Resized image from %dx%d to %dx%d for inference", w, h, new_w, new_h)
        return image, scale

    def _normalize_kp_score(self, value: float, used_scores_tensor: bool) -> float:
        score = float(value)
        if used_scores_tensor:
            if score < 0.0 or score > 1.0:
                score = 1.0 / (1.0 + math.exp(-score))
        else:
            score = max(0.0, min(1.0, score / 2.0))
        if score < 0.0 or score > 1.0:
            logger.debug("Keypoint score out of range after normalization: %.4f", score)
        return max(0.0, min(1.0, score))

    def _estimate_torchvision(self, image: Image.Image) -> List[PersonPose]:
        assert self.model is not None
        weights = self.weights
        orig_w, orig_h = image.size
        image_resized, scale = self._resize_for_inference(image)
        preprocess = weights.transforms() if weights is not None else (lambda im: im)
        tensor = preprocess(image_resized).to(self.inference_device)

        # Ensure model is on the inference device
        if next(self.model.parameters()).device != self.inference_device:
            self.model.to(self.inference_device)

        outputs = self.model([tensor])[0]
        boxes = outputs.get("boxes")
        scores = outputs.get("scores")
        keypoints = outputs.get("keypoints")
        keypoints_scores = outputs.get("keypoints_scores")
        if keypoints_scores is None:
            keypoints_scores = outputs.get("keypoints_score")

        people: List[PersonPose] = []
        if boxes is None or scores is None or keypoints is None:
            logger.debug("No detections from torchvision")
            return people

        num = boxes.shape[0]
        logger.debug("Torchvision detections: %d", int(num))
        inv_scale = (1.0 / scale) if scale != 0 else 1.0
        for i in range(num):
            score = float(scores[i].item())
            if score < self.score_threshold:
                continue
            bbox = boxes[i].tolist()
            if scale != 1.0:
                bbox = [bbox[0] * inv_scale, bbox[1] * inv_scale, bbox[2] * inv_scale, bbox[3] * inv_scale]
            kps = []
            for j in range(keypoints.shape[1]):
                x, y, v = keypoints[i, j].tolist()
                if scale != 1.0:
                    x *= inv_scale
                    y *= inv_scale
                if keypoints_scores is not None:
                    raw = float(keypoints_scores[i, j].item())
                    kp_score = self._normalize_kp_score(raw, used_scores_tensor=True)
                else:
                    kp_score = self._normalize_kp_score(float(v), used_scores_tensor=False)
                name = COCO_KEYPOINT_NAMES[j] if j < len(COCO_KEYPOINT_NAMES) else None
                kps.append(Keypoint(x=float(x), y=float(y), score=kp_score, name=name))
            people.append(PersonPose(keypoints=kps, score=score, bbox=bbox))
        logger.debug("Torchvision kept %d person(s) after threshold", len(people))
        return people
