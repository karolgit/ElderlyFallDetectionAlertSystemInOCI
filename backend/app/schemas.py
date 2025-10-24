from typing import List, Optional
from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    x: float
    y: float
    score: float = Field(0.0, ge=0.0, le=1.0)
    name: Optional[str] = None


class PersonPose(BaseModel):
    keypoints: List[Keypoint]
    score: float = Field(0.0, ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None  # [x1, y1, x2, y2]


class FrameAnalyzeRequest(BaseModel):
    image_base64: str  # data URL or raw base64
    preferred_device: Optional[str] = None  # "mps" | "cuda" | "cpu"


class FrameAnalyzeResponse(BaseModel):
    device: dict
    people: List[PersonPose]
    is_fall: bool
    fall_score: float


class VideoAnalyzeResponse(BaseModel):
    device: dict
    analyzed_frames: int
    any_fall: bool
    fall_frames: List[int]
    average_fall_score: float
