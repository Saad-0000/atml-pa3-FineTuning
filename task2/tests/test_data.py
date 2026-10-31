"""Unit tests for data selection, filtering, and disjointness constraints.
"""
from collections import Counter
import unittest

from task2.src.data.coco_captions import select_coco_captions
from task2.src.data.grounding_pairs import generate_grounding_pair_pools
from task2.src.data.vqav2_pairs import filter_vqav2_complementary_pairs


class TestData(unittest.TestCase):
    def test_coco_caption_prescribed_counts(self):
        """Validates exact caption sample counts: 18,000 train, 96 dev."""
        data = select_coco_captions(seed=6304)
        self.assertEqual(len(data["train_captions"]), 18000)
        self.assertEqual(len(data["train_image_ids"]), 6000)
        self.assertEqual(len(data["dev_captions"]), 96)
        self.assertEqual(len(data["dev_image_ids"]), 96)

        # Verify each train image has exactly 3 captions
        img_counts = Counter(c["image_id"] for c in data["train_captions"])
        self.assertTrue(all(count == 3 for count in img_counts.values()))

    def test_grounding_pair_prescribed_counts(self):
        """Validates Grounding dev (24 pairs / 48 questions) and test (64 pairs / 128 questions)."""
        pools = generate_grounding_pair_pools(seed=6304)
        dev_pairs = pools["dev_pairs"]
        test_pairs = pools["test_pairs"]

        self.assertEqual(len(dev_pairs), 24)
        self.assertEqual(len(test_pairs), 64)

        # Each pair must have different answers
        for p in dev_pairs + test_pairs:
            self.assertNotEqual(p["answer_1"], p["answer_2"])
            self.assertNotEqual(p["image_id_1"], p["image_id_2"])

    def test_vqav2_prescribed_counts_and_consensus(self):
        """Validates natural VQAv2 quotas: train 1536 pairs, dev 32 pairs, test 96 pairs."""
        splits = filter_vqav2_complementary_pairs(seed=6304)
        train_pairs = splits["train_pairs"]
        dev_pairs = splits["dev_pairs"]
        test_pairs = splits["test_pairs"]

        self.assertEqual(len(train_pairs), 1536)
        self.assertEqual(len(dev_pairs), 32)
        self.assertEqual(len(test_pairs), 96)

        # Verify consensus answers differ and annotator agreement >= 8/10
        for p in train_pairs + dev_pairs + test_pairs:
            self.assertNotEqual(p["answer_1"], p["answer_2"])
            self.assertGreaterEqual(p.get("agreement_1", 10), 8)
            self.assertGreaterEqual(p.get("agreement_2", 10), 8)

    def test_train_dev_test_image_disjointness(self):
        """Validates strict mutual image disjointness across train, dev, and test sets."""
        caps = select_coco_captions(seed=6304)
        ground = generate_grounding_pair_pools(seed=6304)
        vqa = filter_vqav2_complementary_pairs(seed=6304)

        train_imgs = set(caps["train_image_ids"]).union(
            {p["image_id_1"] for p in ground["train_presence_pool"] + ground["train_counting_pool"] + vqa["train_pairs"]}
        ).union(
            {p["image_id_2"] for p in ground["train_presence_pool"] + ground["train_counting_pool"] + vqa["train_pairs"]}
        )

        dev_imgs = set(caps["dev_image_ids"]).union(
            {p["image_id_1"] for p in ground["dev_pairs"] + vqa["dev_pairs"]}
        ).union(
            {p["image_id_2"] for p in ground["dev_pairs"] + vqa["dev_pairs"]}
        )

        test_imgs = {p["image_id_1"] for p in ground["test_pairs"] + vqa["test_pairs"]}.union(
            {p["image_id_2"] for p in ground["test_pairs"] + vqa["test_pairs"]}
        )

        self.assertEqual(len(train_imgs.intersection(dev_imgs)), 0, "Train and Dev images overlap!")
        self.assertEqual(len(train_imgs.intersection(test_imgs)), 0, "Train and Test images overlap!")
        self.assertEqual(len(dev_imgs.intersection(test_imgs)), 0, "Dev and Test images overlap!")


if __name__ == "__main__":
    unittest.main()
