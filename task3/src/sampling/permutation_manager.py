"""
Seeded permutation manager for generating reproducible candidate orderings.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class RolloutPermutationManager:
    """Manages generation of 10 deterministic random permutations of 32 rollouts per question."""

    def __init__(
        self,
        base_seed: int = 6304,
        seed: Optional[int] = None,
        num_permutations: int = 10,
        total_rollouts: int = 32
    ):
        self.base_seed = seed if seed is not None else base_seed
        self.num_permutations = num_permutations
        self.total_rollouts = total_rollouts
        self._cache: Dict[str, List[List[int]]] = {}

    def get_permutations_for_question(self, question_id: Any, total_rollouts: int = 32) -> List[List[int]]:
        """Generates 10 distinct, reproducible permutations of rollout indices [0..31] for a question."""
        qid_str = str(question_id)
        if qid_str in self._cache:
            return self._cache[qid_str]

        # Deterministic seed derivation from base_seed and hash of qid_str
        import hashlib
        h = int(hashlib.sha256(f"{self.base_seed}_{qid_str}".encode()).hexdigest()[:8], 16)
        
        orders: List[List[int]] = []
        base_indices = list(range(total_rollouts))

        for order_idx in range(self.num_permutations):
            sub_seed = (h + order_idx * 137) % (2**32 - 1)
            rng = np.random.default_rng(sub_seed)
            perm = rng.permutation(base_indices).tolist()
            orders.append(perm)

        self._cache[qid_str] = orders
        return orders

    def get_candidate_prefix(self, question_id: Any, permutation_index: int, budget_n: int) -> List[int]:
        """Returns the nested prefix of size budget_n from the specified permutation order."""
        orders = self.get_permutations_for_question(question_id, self.total_rollouts)
        p_idx = permutation_index % len(orders)
        return orders[p_idx][:budget_n]


# Alias for compatibility
PermutationManager = RolloutPermutationManager
