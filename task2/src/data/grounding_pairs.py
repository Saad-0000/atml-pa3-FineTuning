"""Programmatic generation of presence and counting grounding complementary pairs from COCO instances."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger("task2.data.grounding_pairs")


def load_coco_instances(instances_file: Path | str) -> Tuple[Dict[int, str], Dict[int, Dict[int, int]], Set[int]]:
    """
    Parses COCO instances json.
    Returns:
        categories: dict mapping category_id -> category_name
        image_cat_counts: dict mapping image_id -> {category_id: count}
        all_image_ids: set of all image_ids
    """
    with open(instances_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    all_image_ids = {img["id"] for img in data.get("images", [])}

    image_cat_counts: Dict[int, Dict[int, int]] = {img_id: {} for img_id in all_image_ids}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in image_cat_counts:
            image_cat_counts[img_id][cat_id] = image_cat_counts[img_id].get(cat_id, 0) + 1

    return categories, image_cat_counts, all_image_ids


def generate_presence_pairs(
    image_ids: List[int],
    categories: Dict[int, str],
    image_cat_counts: Dict[int, Dict[int, int]],
    target_num_pairs: int,
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Generates presence complementary pairs:
    Question: 'Does this image contain [object]?'
    Image A: contains category (target 'yes')
    Image B: does NOT contain category (target 'no')
    """
    rng = np.random.default_rng(seed)
    pairs: List[Dict[str, Any]] = []

    cat_ids = sorted(categories.keys())
    # Group images by category presence
    cat_to_pos_imgs: Dict[int, List[int]] = {c: [] for c in cat_ids}
    cat_to_neg_imgs: Dict[int, List[int]] = {c: [] for c in cat_ids}

    for img_id in image_ids:
        counts = image_cat_counts.get(img_id, {})
        for c in cat_ids:
            if counts.get(c, 0) > 0:
                cat_to_pos_imgs[c].append(img_id)
            else:
                cat_to_neg_imgs[c].append(img_id)

    # Eligible categories with both positive and negative images
    valid_cats = [c for c in cat_ids if cat_to_pos_imgs[c] and cat_to_neg_imgs[c]]
    if not valid_cats:
        return []

    used_img_pairs = set()
    shuffled_cats = rng.permutation(valid_cats).tolist()

    idx = 0
    while len(pairs) < target_num_pairs and idx < target_num_pairs * 50:
        idx += 1
        cat_id = shuffled_cats[idx % len(shuffled_cats)]
        cat_name = categories[cat_id]
        pos_img = rng.choice(cat_to_pos_imgs[cat_id])
        neg_img = rng.choice(cat_to_neg_imgs[cat_id])

        if pos_img == neg_img or (pos_img, neg_img) in used_img_pairs:
            continue

        used_img_pairs.add((pos_img, neg_img))
        pair_id = f"presence_{split_name}_{len(pairs):04d}"
        q_text = f"Does this image contain a {cat_name}?"

        pairs.append({
            "pair_id": pair_id,
            "pair_type": "presence",
            "category_id": cat_id,
            "category_name": cat_name,
            "question": q_text,
            "split": split_name,
            "image_a": {
                "image_id": int(pos_img),
                "answer": "yes",
                "label": 1,
            },
            "image_b": {
                "image_id": int(neg_img),
                "answer": "no",
                "label": 0,
            },
        })

    return pairs


def generate_counting_pairs(
    image_ids: List[int],
    categories: Dict[int, str],
    image_cat_counts: Dict[int, Dict[int, int]],
    target_num_pairs: int,
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Generates counting complementary pairs:
    Question: 'How many [objects] can you see?'
    Pairs images with different counts in range 1-4.
    """
    rng = np.random.default_rng(seed)
    pairs: List[Dict[str, Any]] = []

    cat_ids = sorted(categories.keys())
    # Group images by category and count (1 to 4)
    cat_count_imgs: Dict[int, Dict[int, List[int]]] = {c: {k: [] for k in range(1, 5)} for c in cat_ids}

    for img_id in image_ids:
        counts = image_cat_counts.get(img_id, {})
        for c in cat_ids:
            cnt = counts.get(c, 0)
            if 1 <= cnt <= 4:
                cat_count_imgs[c][cnt].append(img_id)

    # Eligible categories with images in at least two different count brackets in 1-4
    valid_cats = [
        c for c in cat_ids
        if sum(1 for cnt in range(1, 5) if cat_count_imgs[c][cnt]) >= 2
    ]
    if not valid_cats:
        return []

    used_img_pairs = set()
    shuffled_cats = rng.permutation(valid_cats).tolist()

    idx = 0
    while len(pairs) < target_num_pairs and idx < target_num_pairs * 50:
        idx += 1
        cat_id = shuffled_cats[idx % len(shuffled_cats)]
        cat_name = categories[cat_id]

        available_counts = [cnt for cnt in range(1, 5) if cat_count_imgs[cat_id][cnt]]
        if len(available_counts) < 2:
            continue

        c1, c2 = rng.choice(available_counts, size=2, replace=False)
        img1 = rng.choice(cat_count_imgs[cat_id][c1])
        img2 = rng.choice(cat_count_imgs[cat_id][c2])

        if img1 == img2 or (img1, img2) in used_img_pairs:
            continue

        used_img_pairs.add((img1, img2))
        pair_id = f"counting_{split_name}_{len(pairs):04d}"
        plural_name = cat_name if cat_name.endswith("s") else f"{cat_name}s"
        q_text = f"How many {plural_name} can you see?"

        pairs.append({
            "pair_id": pair_id,
            "pair_type": "counting",
            "category_id": cat_id,
            "category_name": cat_name,
            "question": q_text,
            "split": split_name,
            "image_a": {
                "image_id": int(img1),
                "answer": str(c1),
                "count": int(c1),
            },
            "image_b": {
                "image_id": int(img2),
                "answer": str(c2),
                "count": int(c2),
            },
        })

    return pairs


def build_grounding_dataset(
    val_instances_json: Path | str,
    train_instances_json: Optional[Path | str] = None,
    excluded_image_ids: Optional[Set[int]] = None,
    seed: int = 6304,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds:
    - Development grounding suite: 24 pairs / 48 questions (12 presence, 12 counting) from val2014
    - Final test grounding suite: 64 pairs / 128 questions (32 presence, 32 counting) from val2014
    - Train pools (presence & counting) from train2014 (or non-overlapping val2014 if train instances provided)
    Ensures complete image disjointness between Dev and Final Test!
    """
    rng = np.random.default_rng(seed)
    excluded = set(excluded_image_ids or [])

    val_cats, val_counts, val_all_imgs = load_coco_instances(val_instances_json)
    eligible_val_imgs = sorted(list(val_all_imgs - excluded))

    # Permute eligible val images to split into disjoint Dev and Final-test candidate pools
    shuffled_val = rng.permutation(eligible_val_imgs).tolist()
    split_point = len(shuffled_val) // 2
    dev_pool_imgs = shuffled_val[:split_point]
    test_pool_imgs = shuffled_val[split_point:]

    # 1. Dev suite: 12 presence pairs + 12 counting pairs = 24 pairs (48 questions)
    dev_presence = generate_presence_pairs(dev_pool_imgs, val_cats, val_counts, 12, seed, "dev")
    dev_counting = generate_counting_pairs(dev_pool_imgs, val_cats, val_counts, 12, seed + 1, "dev")
    dev_pairs = dev_presence + dev_counting

    # Record all image IDs used in dev
    dev_used_imgs = set()
    for p in dev_pairs:
        dev_used_imgs.add(p["image_a"]["image_id"])
        dev_used_imgs.add(p["image_b"]["image_id"])

    # 2. Final test suite: 32 presence pairs + 32 counting pairs = 64 pairs (128 questions)
    # Strictly exclude any dev images!
    clean_test_pool = [img for img in test_pool_imgs if img not in dev_used_imgs]
    test_presence = generate_presence_pairs(clean_test_pool, val_cats, val_counts, 32, seed + 2, "final_test")
    test_counting = generate_counting_pairs(clean_test_pool, val_cats, val_counts, 32, seed + 3, "final_test")
    final_test_pairs = test_presence + test_counting

    # 3. Train grounding pools from train2014
    train_pairs: List[Dict[str, Any]] = []
    if train_instances_json and Path(train_instances_json).exists():
        tr_cats, tr_counts, tr_all_imgs = load_coco_instances(train_instances_json)
        # Exclude any dev/test images
        test_used_imgs = set()
        for p in final_test_pairs:
            test_used_imgs.add(p["image_a"]["image_id"])
            test_used_imgs.add(p["image_b"]["image_id"])
        clean_tr_imgs = sorted(list(tr_all_imgs - dev_used_imgs - test_used_imgs))
        tr_presence = generate_presence_pairs(clean_tr_imgs, tr_cats, tr_counts, 2000, seed + 4, "train")
        tr_counting = generate_counting_pairs(clean_tr_imgs, tr_cats, tr_counts, 2000, seed + 5, "train")
        train_pairs = tr_presence + tr_counting

    manifest = {
        "dev_pairs_count": len(dev_pairs),
        "dev_questions_count": len(dev_pairs) * 2,
        "final_test_pairs_count": len(final_test_pairs),
        "final_test_questions_count": len(final_test_pairs) * 2,
        "train_pool_pairs_count": len(train_pairs),
        "dev_image_count": len(dev_used_imgs),
    }

    logger.info(
        f"Grounding suites created: Dev {len(dev_pairs)} pairs ({len(dev_pairs)*2} questions), "
        f"Final Test {len(final_test_pairs)} pairs ({len(final_test_pairs)*2} questions)."
    )
    return dev_pairs, final_test_pairs, train_pairs, manifest


def generate_grounding_pair_pools(
    instances_val_json: Optional[Path | str] = None,
    instances_train_json: Optional[Path | str] = None,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Generates grounding pairs (dev: 24 pairs, test: 64 pairs, and train pools)."""
    if instances_val_json and Path(instances_val_json).exists():
        dev_p, test_p, train_p, manifest = build_grounding_dataset(
            val_instances_json=instances_val_json,
            train_instances_json=instances_train_json,
            seed=seed,
        )
        # Flatten helper
        def flatten_pair(p):
            return {
                "pair_id": p["pair_id"],
                "question": p["question"],
                "image_id_1": p["image_a"]["image_id"],
                "image_id_2": p["image_b"]["image_id"],
                "answer_1": p["image_a"]["answer"],
                "answer_2": p["image_b"]["answer"],
                "pair_type": p.get("pair_type", "presence"),
                "image_a": p["image_a"],
                "image_b": p["image_b"],
            }
        return {
            "dev_pairs": [flatten_pair(p) for p in dev_p],
            "test_pairs": [flatten_pair(p) for p in test_p],
            "train_presence_pool": [flatten_pair(p) for p in train_p if p.get("pair_type") == "presence"],
            "train_counting_pool": [flatten_pair(p) for p in train_p if p.get("pair_type") == "counting"],
        }

    # Deterministic generation for offline testing / dry-run
    rng = np.random.default_rng(seed)
    categories = ["cat", "dog", "car", "chair", "person", "airplane", "boat", "bicycle"]

    # Dev: 24 pairs (12 presence, 12 counting)
    dev_pairs = []
    for i in range(12):
        cat = categories[i % len(categories)]
        img1 = 300000000000 + i * 2 + 1
        img2 = 300000000000 + i * 2 + 2
        dev_pairs.append({
            "pair_id": f"dev_presence_{i+1}",
            "question": f"Does this image contain a {cat}?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "yes",
            "answer_2": "no",
            "pair_type": "presence",
        })
    for i in range(12):
        cat = categories[i % len(categories)]
        img1 = 300000000025 + i * 2 + 1
        img2 = 300000000025 + i * 2 + 2
        dev_pairs.append({
            "pair_id": f"dev_counting_{i+1}",
            "question": f"How many {cat}s can you see?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "1",
            "answer_2": "2",
            "pair_type": "counting",
        })

    # Test: 64 pairs (32 presence, 32 counting)
    test_pairs = []
    for i in range(32):
        cat = categories[i % len(categories)]
        img1 = 400000000000 + i * 2 + 1
        img2 = 400000000000 + i * 2 + 2
        test_pairs.append({
            "pair_id": f"test_presence_{i+1}",
            "question": f"Does this image contain a {cat}?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "yes",
            "answer_2": "no",
            "pair_type": "presence",
        })
    for i in range(32):
        cat = categories[i % len(categories)]
        img1 = 400000000065 + i * 2 + 1
        img2 = 400000000065 + i * 2 + 2
        test_pairs.append({
            "pair_id": f"test_counting_{i+1}",
            "question": f"How many {cat}s can you see?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "1",
            "answer_2": "3",
            "pair_type": "counting",
        })

    # Training pools
    train_presence = []
    for i in range(500):
        cat = categories[i % len(categories)]
        img1 = 500000000000 + i * 2 + 1
        img2 = 500000000000 + i * 2 + 2
        train_presence.append({
            "pair_id": f"train_presence_{i+1}",
            "question": f"Does this image contain a {cat}?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "yes",
            "answer_2": "no",
            "pair_type": "presence",
        })

    train_counting = []
    for i in range(500):
        cat = categories[i % len(categories)]
        img1 = 600000000000 + i * 2 + 1
        img2 = 600000000000 + i * 2 + 2
        train_counting.append({
            "pair_id": f"train_counting_{i+1}",
            "question": f"How many {cat}s can you see?",
            "image_id_1": img1,
            "image_id_2": img2,
            "answer_1": "2",
            "answer_2": "4",
            "pair_type": "counting",
        })

    return {
        "dev_pairs": dev_pairs,
        "test_pairs": test_pairs,
        "train_presence_pool": train_presence,
        "train_counting_pool": train_counting,
    }
