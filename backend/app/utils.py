import base64
import io
from typing import Tuple
from PIL import Image

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def data_url_to_image(data_url: str) -> Image.Image:
    """Converts a base64 data URL or raw base64 string to a PIL Image (RGB)."""
    if "," in data_url and data_url.strip().lower().startswith("data:"):
        base64_part = data_url.split(",", 1)[1]
    else:
        base64_part = data_url
    binary = base64.b64decode(base64_part)
    image = Image.open(io.BytesIO(binary)).convert("RGB")
    return image


def pil_to_cv2_rgb(image: Image.Image):
    import numpy as np

    return np.array(image)
