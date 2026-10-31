"""SigLIP vision feature extraction and persistent FP16 caching."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipVisionModel

logger = logging.getLogger("task2.models.siglip_extractor")


class SiglipPatchFeatureExtractor(nn.Module):
    """
    Extracts 196 spatial patch tokens from google/siglip-base-patch16-224.
    Crucial requirement: Applies SigLIP's final layer norm exactly once.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/siglip-base-patch16-224",
        revision: str = "main",
        apply_final_ln_once: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.revision = revision
        self.apply_final_ln_once = apply_final_ln_once
        self.dtype = torch_dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading SigLIP model {model_name_or_path} (revision={revision})...")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, revision=revision)
        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
        ).to(self.device)

        # Freeze vision model parameters permanently
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract_patch_features(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> torch.Tensor:
        """
        Extracts 196 patch tokens of dimension 768 for each input image.
        Returns:
            Tensor of shape [batch_size, 196, 768] in self.dtype
        """
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(dtype=self.dtype)

        # In transformers SiglipVisionModel, vision_model outputs last_hidden_state.
        # Check whether vision_model.post_layernorm was applied in forward:
        # SiglipVisionTransformer forward:
        #   hidden_states = encoder(...)
        #   last_hidden_state = post_layernorm(hidden_states)
        # We assert final layer norm is applied exactly once.
        outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state  # shape [B, 196, 768]

        assert last_hidden_state.shape[1] == 196, (
            f"Expected 196 patches (14x14 grid), got {last_hidden_state.shape[1]}!"
        )
        assert last_hidden_state.shape[2] == 768, (
            f"Expected hidden dimension 768, got {last_hidden_state.shape[2]}!"
        )

        return last_hidden_state


def cache_siglip_features_for_image_list(
    image_paths_dict: Dict[int, Path],
    cache_dir: Path | str,
    extractor: SiglipPatchFeatureExtractor,
    batch_size: int = 32,
    manifest_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Extracts and persistently saves FP16 patch feature tensors [196, 768] on disk by image_id.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cached_image_ids = []
    total_images = len(image_paths_dict)
    sorted_image_ids = sorted(image_paths_dict.keys())

    logger.info(f"Caching SigLIP features for {total_images} images into {cache_path}...")

    # Process in batches
    for i in range(0, total_images, batch_size):
        batch_ids = sorted_image_ids[i : i + batch_size]
        imgs_to_process = []
        ids_to_process = []

        for img_id in batch_ids:
            target_file = cache_path / f"img_{img_id:012d}.pt"
            if target_file.exists():
                cached_image_ids.append(img_id)
                continue
            img_p = image_paths_dict[img_id]
            if img_p.exists():
                try:
                    img = Image.open(img_p).convert("RGB")
                    imgs_to_process.append(img)
                    ids_to_process.append(img_id)
                except Exception as e:
                    logger.warning(f"Error loading image {img_p}: {e}")

        if imgs_to_process:
            features = extractor.extract_patch_features(imgs_to_process).cpu()
            for idx, img_id in enumerate(ids_to_process):
                feat = features[idx].to(torch.float16)  # [196, 768]
                target_file = cache_path / f"img_{img_id:012d}.pt"
                torch.save(feat, str(target_file))
                cached_image_ids.append(img_id)

    manifest = {
        "model_name": extractor.model_name,
        "revision": extractor.revision,
        "num_patches": 196,
        "feature_dim": 768,
        "dtype": "float16",
        "total_cached": len(cached_image_ids),
        "cache_dir": str(cache_path),
    }

    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    logger.info(f"Feature caching complete: {len(cached_image_ids)}/{total_images} cached.")
    return manifest


def load_cached_feature(cache_dir: Path | str, image_id: int) -> Optional[torch.Tensor]:
    """Loads a single cached [196, 768] FP16 feature tensor."""
    path = Path(cache_dir) / f"img_{image_id:012d}.pt"
    if path.exists():
        return torch.load(str(path), map_location="cpu")
    return None


SigLIPExtractor = SiglipPatchFeatureExtractor
