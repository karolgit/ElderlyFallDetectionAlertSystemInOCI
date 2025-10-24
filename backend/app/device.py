import os
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def get_torch_device(preferred: str | None = None) -> Tuple[torch.device, str]:
    preferred_normalized = (preferred or "").strip().lower()

    if preferred_normalized in {"mps", "cuda", "cpu"}:
        device_type = preferred_normalized
        logger.debug("Preferred device requested: %s", device_type)
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_type = "mps"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
        logger.debug("Auto-selected device: %s", device_type)

    device = torch.device(device_type)
    return device, device_type


def summarize_device(device: torch.device) -> dict:
    info: dict[str, object] = {"type": device.type}
    if device.type == "cuda":
        try:
            idx = torch.cuda.current_device()
            info["name"] = torch.cuda.get_device_name(idx)
            info["capability"] = torch.cuda.get_device_capability(idx)
        except Exception as e:
            logger.debug("CUDA device info error: %s", e)
            info["name"] = "unknown"
    elif device.type == "mps":
        info["name"] = "Apple Metal (MPS)"
    else:
        info["name"] = "CPU"
    return info
