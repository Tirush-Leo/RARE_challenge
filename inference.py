# inference.py
"""
Container entrypoint:
- Resolves interface via /input/inputs.json
- Loads FIRST .tif/.tiff/.mha from /input/images/stacked-barretts-esophagus-endoscopy
- Splits stacks into frames
- Runs ViT-based model (with CLAHE + CenterCrop) from resources/rare_vit_best.pth
- Writes /output/stacked-neoplastic-lesion-likelihoods.json (list of probabilities)
"""

from pathlib import Path
import json
from glob import glob
from typing import List

import numpy as np
import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    interface_key = get_interface_key()
    handlers = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]
    return handlers()


def interface_0_handler():
    # 1) Read inputs (stack → list of frames)
    frames = load_images_as_list(location=INPUT_PATH / "images" / "stacked-barretts-esophagus-endoscopy")

    # 2) Optional CUDA info
    _show_torch_cuda_info()

    # 3) Load your new model (keeps the baseline structure)
    from model.rare_vit_model import RareVitModel
    print("RARE ViT model (CLAHE + CenterCrop)")
    model = RareVitModel(
        weights=RESOURCE_PATH / "rare_vit_best.pth",
        model_name="vit_base_patch16_224",
        img_size=224,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        clahe_colorspace="YUV",
    )

    # 4) Predict
    probs = model.predict(frames)

    # 5) Save output
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=probs,
    )

    return 0


# ---------------------------
# Helpers (I/O + plumbing)
# ---------------------------
def get_interface_key() -> tuple:
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location: Path):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location: Path, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_images_as_list(*, location):
    """
    Returns a list of frames from the FIRST supported file in `location`.
    Supports .tif/.tiff/.mha, shapes: [Z,H,W], [Z,H,W,C], [H,W,C], [H,W].
    """
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    img = sitk.ReadImage(input_files[0])
    return sitk.GetArrayFromImage(img)  # [Z,H,W,(C?)] or [H,W,(C)]



def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    available = torch.cuda.is_available()
    print(f"Torch CUDA is available: {available}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"\tcurrent device: {current_device}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
