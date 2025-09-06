# model/rare_vit_model.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import cv2


def apply_clahe(
    img: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    colorspace: str = "YUV",  # "YUV" or "LAB"
) -> Image.Image:
    """CLAHE on luminance channel; returns RGB PIL.Image (uint8)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_np = np.asarray(img)  # HxWx3 (uint8)

    cs = colorspace.upper()
    if cs == "YUV":
        code_from, code_to, l_idx = cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0
    elif cs == "LAB":
        code_from, code_to, l_idx = cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB, 0
    else:
        raise ValueError("colorspace must be 'YUV' or 'LAB'")

    img_cs = cv2.cvtColor(img_np, code_from)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    img_cs[..., l_idx] = clahe.apply(img_cs[..., l_idx])
    img_out = cv2.cvtColor(img_cs, code_to)
    return Image.fromarray(img_out)


class ViTClassifier(nn.Module):
    """
    ViT backbone (features) + small MLP head → 2-class logits.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 2):
        super().__init__()
        # Use num_classes=0 to get features; set pretrained=False (we load our own weights)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        embed_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, 128),       nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)           # [B, embed_dim]
        logits = self.classifier(feat)    # [B, 2]
        return logits


class RareVitModel:
    """
    Wrapper that:
      - builds ViTClassifier
      - loads weights (supports {'model': state_dict} or raw state_dict)
      - applies preprocessing (CLAHE -> Resize -> CenterCrop -> ToTensor -> Normalize)
      - predicts class-1 probabilities via softmax
    """
    def __init__(
        self,
        weights: Path,
        model_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        device: Optional[torch.device] = None,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        clahe_colorspace: str = "YUV",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTClassifier(model_name=model_name, num_classes=2).to(self.device).eval()

        ckpt = torch.load(weights, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)

        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe_colorspace = clahe_colorspace

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: apply_clahe(
                img,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid_size,
                colorspace=self.clahe_colorspace,
            )) if self.use_clahe else transforms.Lambda(lambda img: img),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        """Convert any array to uint8 sensibly."""
        if np.issubdtype(arr.dtype, np.floating):
            finite = np.isfinite(arr)
            if not finite.any():
                return np.zeros_like(arr, dtype=np.uint8)
            amin, amax = float(arr[finite].min()), float(arr[finite].max())
            if amin >= 0 and amax <= 1:
                arr = arr * 255.0
            elif not (amin >= 0 and amax <= 255):
                arr = np.zeros_like(arr) if amax == amin else (arr - amin) * (255.0 / (amax - amin))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _to_pil(arr: np.ndarray) -> Image.Image:
        """Ensure HWC uint8 RGB PIL.Image."""
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
        else:
            raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = RareVitModel._to_uint8(arr)
        return Image.fromarray(arr, mode="RGB")

    @torch.no_grad()
    def predict(self, images: List[np.ndarray]) -> List[float]:
        """Return per-frame class-1 probabilities (softmax logits[:,1])."""
        probs: List[float] = []
        for arr in images:
            pil = self._to_pil(arr)
            x = self.transform(pil).unsqueeze(0).to(self.device)
            logits = self.model(x)                 # [1,2]
            p1 = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(float(p1))
        return probs
