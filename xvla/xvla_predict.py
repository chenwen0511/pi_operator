import torch
import torch.nn as nn
from typing import Dict, Any, Optional


MODEL_REPO = "lerobot/xvla-folding"
LOCAL_PATH = "/home/ubuntu/stephen/02-weight/xvla-folding"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_preprocessor_config():
    return {
        "max_length": 50,
        "task_key": "task",
        "padding_side": "right",
        "padding": "max_length",
        "truncation": True,
        "tokenizer_name": "facebook/bart-large",
    }


def get_postprocessor_config():
    return {
        "eps": 1e-8,
    }


class XVLAFoldingInferencer:
    def __init__(
        self,
        model_path: str = LOCAL_PATH,
        device: str = DEVICE,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        try:
            from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
            self.policy = XVLAPolicy.from_pretrained(model_path)
            self.policy.to(device=device, dtype=dtype)
            self.policy.eval()
        except ImportError:
            self.policy = None
            print("Warning: LeRobot not installed. Using mock policy for structure.")

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch]) -> torch.Tensor:
        if self.policy is None:
            return torch.randn(1, 20, device=self.device, dtype=self.dtype)
        return self.policy.select_action(batch)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch]) -> Dict[str, Any]:
        if self.policy is None:
            return {"loss": None, "action": torch.randn(1, 20, device=self.device, dtype=self.dtype)}
        return self.policy.forward(batch)


def normalize_observation(
    images: Dict[str, torch.Tensor],
    states: torch.Tensor,
    mean: Optional[Dict[str, torch.Tensor]] = None,
    std: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    normalized = {}
    
    for key, img in images.items():
        normalized[key] = img / 255.0 * 2.0 - 1.0
    
    if mean is not None and std is not None:
        normalized["state"] = (states - mean) / (std + 1e-8)
    else:
        normalized["state"] = states
    
    return normalized


def unnormalize_action(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return action * std + mean


def preprocess_batch(
    images: Dict[str, torch.Tensor],
    state: torch.Tensor,
    task: Optional[str] = None,
    image_size: tuple = (256, 256),
) -> Dict[str, Any]:
    batch = {}
    
    for key, img in images.items():
        if img.shape[-2:] != image_size:
            img = torch.nn.functional.interpolate(
                img,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
        batch[key] = img
    
    batch["observation.state"] = state
    
    if task is not None:
        batch["task"] = task
    
    return batch


def resize_with_padding(
    image: torch.Tensor,
    target_size: tuple = (224, 224),
) -> torch.Tensor:
    _, C, H, W = image.shape
    target_h, target_w = target_size
    
    scale = min(target_h / H, target_w / W)
    new_h, new_w = int(H * scale), int(W * scale)
    
    resized = torch.nn.functional.interpolate(
        image,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    
    padded = torch.zeros(C, target_h, target_w)
    h_start = (target_h - new_h) // 2
    w_start = (target_w - new_w) // 2
    padded[:, h_start:h_start+new_h, w_start:w_start+new_w] = resized
    
    return padded


if __name__ == "__main__":
    inferencer = XVLAFoldingInferencer()
    
    batch = {
        "observation.images.image": torch.randn(1, 3, 256, 256).to(DEVICE),
        "observation.images.image2": torch.randn(1, 3, 256, 256).to(DEVICE),
        "observation.images.image3": torch.randn(1, 3, 224, 224).to(DEVICE),
        "observation.state": torch.randn(1, 8).to(DEVICE),
        "task": "fold the cloth",
    }
    action = inferencer.select_action(batch)
    print(f"Action shape: {action.shape}")