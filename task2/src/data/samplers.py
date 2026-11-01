"""Seeded mixture samplers and batch collators for the 3 curriculum training stages."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StageMixtureSampler:
    """
    Manages deterministic curriculum sampling across stages with tracked mixture counts.
    Each draw contributes exactly 2 examples (1 complementary pair or 2 replay captions).
    """

    def __init__(
        self,
        seed: int = 6304,
        stage_index: int = 1,
        caption_pool: Optional[List[Dict[str, Any]]] = None,
        presence_pool: Optional[List[Dict[str, Any]]] = None,
        counting_pool: Optional[List[Dict[str, Any]]] = None,
        vqav2_pool: Optional[List[Dict[str, Any]]] = None,
    ):
        self.seed = seed
        self.stage_index = stage_index
        self.rng = np.random.default_rng(seed)

        self.caption_pool = caption_pool or []
        self.presence_pool = presence_pool or []
        self.counting_pool = counting_pool or []
        self.vqav2_pool = vqav2_pool or []

        self.mixture_counts: Dict[str, int] = {
            "caption": 0,
            "presence": 0,
            "counting": 0,
            "vqav2": 0,
        }
        self.total_draws = 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "stage_index": self.stage_index,
            "rng_state": self.rng.bit_generator.state,
            "mixture_counts": dict(self.mixture_counts),
            "total_draws": self.total_draws,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.seed = state.get("seed", self.seed)
        self.stage_index = state.get("stage_index", self.stage_index)
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        self.mixture_counts = dict(state.get("mixture_counts", self.mixture_counts))
        self.total_draws = state.get("total_draws", self.total_draws)

    def sample_draw(self) -> List[Dict[str, Any]]:
        return self.draw_pair_or_two_examples()

    def draw_pair_or_two_examples(self) -> List[Dict[str, Any]]:
        """
        Draws 2 examples according to the current stage sampling policy.
        """
        self.total_draws += 1

        if self.stage_index == 1:
            # Stage 1: pure caption alignment (draw 2 captions)
            self.mixture_counts["caption"] += 2
            if not self.caption_pool:
                return []
            idx1, idx2 = self.rng.choice(len(self.caption_pool), size=2, replace=False)
            return [self.caption_pool[idx1], self.caption_pool[idx2]]

        elif self.stage_index == 2:
            # Stage 2: 50% presence, 25% counting, 25% caption replay
            p = self.rng.random()
            if p < 0.50 and self.presence_pool:
                self.mixture_counts["presence"] += 2
                pair = self.presence_pool[self.rng.integers(len(self.presence_pool))]
                return self._pair_to_examples(pair)
            elif p < 0.75 and self.counting_pool:
                self.mixture_counts["counting"] += 2
                pair = self.counting_pool[self.rng.integers(len(self.counting_pool))]
                return self._pair_to_examples(pair)
            else:
                self.mixture_counts["caption"] += 2
                if self.caption_pool:
                    idx1, idx2 = self.rng.choice(len(self.caption_pool), size=2, replace=False)
                    return [self.caption_pool[idx1], self.caption_pool[idx2]]
                return []

        elif self.stage_index == 3:
            # Stage 3: 70% VQAv2, 10% presence, 10% counting, 10% caption replay
            p = self.rng.random()
            if p < 0.70 and self.vqav2_pool:
                self.mixture_counts["vqav2"] += 2
                pair = self.vqav2_pool[self.rng.integers(len(self.vqav2_pool))]
                return self._pair_to_examples(pair)
            elif p < 0.80 and self.presence_pool:
                self.mixture_counts["presence"] += 2
                pair = self.presence_pool[self.rng.integers(len(self.presence_pool))]
                return self._pair_to_examples(pair)
            elif p < 0.90 and self.counting_pool:
                self.mixture_counts["counting"] += 2
                pair = self.counting_pool[self.rng.integers(len(self.counting_pool))]
                return self._pair_to_examples(pair)
            else:
                self.mixture_counts["caption"] += 2
                if self.caption_pool:
                    idx1, idx2 = self.rng.choice(len(self.caption_pool), size=2, replace=False)
                    return [self.caption_pool[idx1], self.caption_pool[idx2]]
                return []

        return []

    def draw_effective_batch(self, effective_batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Draws effective batch (16 examples = 8 pair draws).
        """
        batch: List[Dict[str, Any]] = []
        num_pair_draws = effective_batch_size // 2
        for _ in range(num_pair_draws):
            batch.extend(self.draw_pair_or_two_examples())
        return batch

    @staticmethod
    def _pair_to_examples(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts a complementary pair dict into 2 training example dicts.
        Supports both nested schema (image_a, image_b) and flat schema (image_id_1, image_id_2).
        """
        q = pair.get("question", "")
        pair_id = pair.get("pair_id", "pair")
        pair_type = pair.get("pair_type", pair.get("answer_type", "vqa"))

        if "image_a" in pair and "image_b" in pair and isinstance(pair["image_a"], dict):
            img_a = pair["image_a"]
            img_b = pair["image_b"]
            img_id_a = img_a.get("image_id")
            img_id_b = img_b.get("image_id")
            ans_a = img_a.get("answer") or img_a.get("consensus_answer", "")
            ans_b = img_b.get("answer") or img_b.get("consensus_answer", "")
        else:
            img_id_a = pair.get("image_id_1") or pair.get("image_id_a")
            img_id_b = pair.get("image_id_2") or pair.get("image_id_b")
            ans_a = pair.get("answer_1") or pair.get("answer_a", "")
            ans_b = pair.get("answer_2") or pair.get("answer_b", "")

        ex_a = {
            "example_id": f"{pair_id}_a",
            "pair_id": pair_id,
            "pair_role": "image_a",
            "image_id": img_id_a,
            "question": q,
            "answer": str(ans_a),
            "target_text": str(ans_a),
            "component": pair_type,
            "is_caption": False,
        }
        ex_b = {
            "example_id": f"{pair_id}_b",
            "pair_id": pair_id,
            "pair_role": "image_b",
            "image_id": img_id_b,
            "question": q,
            "answer": str(ans_b),
            "target_text": str(ans_b),
            "component": pair_type,
            "is_caption": False,
        }
        return [ex_a, ex_b]


class Stage2MixtureSampler(StageMixtureSampler):
    """Stage 2: 50% presence, 25% counting, 25% caption replay."""

    def __init__(
        self,
        presence_pool: Optional[List[Dict[str, Any]]] = None,
        counting_pool: Optional[List[Dict[str, Any]]] = None,
        caption_pool: Optional[List[Dict[str, Any]]] = None,
        seed: int = 6306,
    ):
        super().__init__(
            seed=seed,
            stage_index=2,
            presence_pool=presence_pool,
            counting_pool=counting_pool,
            caption_pool=caption_pool,
        )


class Stage3MixtureSampler(StageMixtureSampler):
    """Stage 3: 70% VQAv2, 10% presence, 10% counting, 10% caption replay."""

    def __init__(
        self,
        vqav2_pool: Optional[List[Dict[str, Any]]] = None,
        presence_pool: Optional[List[Dict[str, Any]]] = None,
        counting_pool: Optional[List[Dict[str, Any]]] = None,
        caption_pool: Optional[List[Dict[str, Any]]] = None,
        seed: int = 6307,
    ):
        super().__init__(
            seed=seed,
            stage_index=3,
            vqav2_pool=vqav2_pool,
            presence_pool=presence_pool,
            counting_pool=counting_pool,
            caption_pool=caption_pool,
        )
