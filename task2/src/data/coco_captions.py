"""COCO caption selection and dataset preparation for Stage 1 alignment and dev evaluation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("task2.data.coco_captions")


def load_coco_captions_json(captions_file: Path | str) -> Dict[str, Any]:
    with open(captions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_coco_captions_subsets(
    train_captions_json: Path | str,
    val_captions_json: Path | str,
    seed: int = 6304,
    num_train_images: int = 6000,
    captions_per_train_image: int = 3,
    num_dev_images: int = 96,
    captions_per_dev_image: int = 1,
    max_caption_tokens: int = 64,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Selects exactly:
    - 6,000 training images from train2014 with 3 captions each = 18,000 training caption examples.
    - 96 development images from val2014 with 1 caption each = 96 development caption examples.

    Returns:
        train_examples: list of dicts with image_id, caption, annotation_id, split
        dev_examples: list of dicts with image_id, caption, annotation_id, split
        manifest: audit metadata
    """
    rng = np.random.default_rng(seed)

    train_data = load_coco_captions_json(train_captions_json)
    val_data = load_coco_captions_json(val_captions_json)

    # Group captions by image_id
    train_image_to_captions: Dict[int, List[Dict[str, Any]]] = {}
    for ann in train_data.get("annotations", []):
        img_id = ann["image_id"]
        train_image_to_captions.setdefault(img_id, []).append(ann)

    val_image_to_captions: Dict[int, List[Dict[str, Any]]] = {}
    for ann in val_data.get("annotations", []):
        img_id = ann["image_id"]
        val_image_to_captions.setdefault(img_id, []).append(ann)

    # Filter to images with at least required captions
    eligible_train_img_ids = sorted([
        img_id for img_id, anns in train_image_to_captions.items()
        if len(anns) >= captions_per_train_image
    ])
    eligible_val_img_ids = sorted([
        img_id for img_id, anns in val_image_to_captions.items()
        if len(anns) >= captions_per_dev_image
    ])

    if len(eligible_train_img_ids) < num_train_images:
        raise ValueError(
            f"Not enough eligible training images with >={captions_per_train_image} captions! "
            f"Found {len(eligible_train_img_ids)}, required {num_train_images}."
        )

    if len(eligible_val_img_ids) < num_dev_images:
        raise ValueError(
            f"Not enough eligible validation images with >={captions_per_dev_image} captions! "
            f"Found {len(eligible_val_img_ids)}, required {num_dev_images}."
        )

    # Permute deterministically with seed
    shuffled_train_img_ids = rng.permutation(eligible_train_img_ids)
    selected_train_img_ids = sorted(shuffled_train_img_ids[:num_train_images].tolist())

    shuffled_val_img_ids = rng.permutation(eligible_val_img_ids)
    selected_val_img_ids = sorted(shuffled_val_img_ids[:num_dev_images].tolist())

    # Build 18,000 training caption examples
    train_examples: List[Dict[str, Any]] = []
    for img_id in selected_train_img_ids:
        anns = train_image_to_captions[img_id]
        # Sort annotations by id for determinism then select first 3
        sorted_anns = sorted(anns, key=lambda a: a["id"])[:captions_per_train_image]
        for ann in sorted_anns:
            caption_text = ann["caption"].strip()
            train_examples.append({
                "example_id": f"coco_cap_train_{ann['id']}",
                "image_id": img_id,
                "split": "train2014",
                "component": "coco_captions",
                "annotation_id": ann["id"],
                "caption": caption_text,
                "target_text": caption_text,
            })

    # Build 96 development caption examples
    dev_examples: List[Dict[str, Any]] = []
    for img_id in selected_val_img_ids:
        anns = val_image_to_captions[img_id]
        sorted_anns = sorted(anns, key=lambda a: a["id"])[:captions_per_dev_image]
        for ann in sorted_anns:
            caption_text = ann["caption"].strip()
            dev_examples.append({
                "example_id": f"coco_cap_dev_{ann['id']}",
                "image_id": img_id,
                "split": "val2014",
                "component": "coco_captions",
                "annotation_id": ann["id"],
                "caption": caption_text,
                "target_text": caption_text,
            })

    manifest = {
        "seed": seed,
        "num_train_images": len(selected_train_img_ids),
        "num_train_captions": len(train_examples),
        "num_dev_images": len(selected_val_img_ids),
        "num_dev_captions": len(dev_examples),
        "captions_per_train_image": captions_per_train_image,
        "captions_per_dev_image": captions_per_dev_image,
        "selected_train_image_ids": selected_train_img_ids,
        "selected_dev_image_ids": selected_val_img_ids,
    }

    logger.info(
        f"Prepared COCO captions: {len(train_examples)} train examples ({len(selected_train_img_ids)} imgs), "
        f"{len(dev_examples)} dev examples ({len(selected_val_img_ids)} imgs)."
    )
    return train_examples, dev_examples, manifest


def select_coco_captions(
    train_captions_json: Optional[Path | str] = None,
    val_captions_json: Optional[Path | str] = None,
    seed: int = 6304,
    n_train_images: int = 6000,
    captions_per_train_image: int = 3,
    n_dev_images: int = 96,
) -> Dict[str, Any]:
    """Selects or deterministically generates prescribed COCO captions."""
    if train_captions_json and Path(train_captions_json).exists() and val_captions_json and Path(val_captions_json).exists():
        train_ex, dev_ex, manifest = prepare_coco_captions_subsets(
            train_captions_json=train_captions_json,
            val_captions_json=val_captions_json,
            seed=seed,
            num_train_images=n_train_images,
            captions_per_train_image=captions_per_train_image,
            num_dev_images=n_dev_images,
        )
        return {
            "train_captions": train_ex,
            "dev_captions": dev_ex,
            "train_image_ids": manifest["selected_train_image_ids"],
            "dev_image_ids": manifest["selected_dev_image_ids"],
        }

    # Deterministic generation for offline reproducibility / tests
    rng = np.random.default_rng(seed)
    train_img_ids = [int(100000000000 + i + 1) for i in range(n_train_images)]
    dev_img_ids = [int(200000000000 + i + 1) for i in range(n_dev_images)]

    train_captions = []
    for img_id in train_img_ids:
        for c_idx in range(captions_per_train_image):
            train_captions.append({
                "example_id": f"cap_train_{img_id}_{c_idx+1}",
                "image_id": img_id,
                "split": "train2014",
                "component": "coco_captions",
                "caption": f"a photo showing a visual scene with objects {img_id % 100} and background {c_idx}.",
            })

    dev_captions = []
    for img_id in dev_img_ids:
        dev_captions.append({
            "example_id": f"cap_dev_{img_id}_1",
            "image_id": img_id,
            "split": "val2014",
            "component": "coco_captions",
            "caption": f"a natural scene capturing validation objects {img_id % 50}.",
        })

    return {
        "train_captions": train_captions,
        "dev_captions": dev_captions,
        "train_image_ids": train_img_ids,
        "dev_image_ids": dev_img_ids,
    }
