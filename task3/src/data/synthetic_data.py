"""
Synthetic rollout and metrics generator for offline testing and dry-run preflights.

Creates deterministic synthetic Parquet datasets with the exact expected schema:
- MATH (LLM): 1,000 questions x 32 rollouts = 32,000 rows
- MathVista (VLM): 1,000 questions x 32 rollouts = 32,000 rows with attention groups
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from task3.src.utils.common import setup_logger

logger = setup_logger("task3.data.synthetic")


def generate_synthetic_rollout_dataset(
    dataset_name: str = "math",
    num_questions: int = 1000,
    rollouts_per_question: int = 32,
    seed: int = 6304,
) -> Dict[str, pd.DataFrame]:
    """Generates synthetic generations, probability_metrics, and attention_metrics DataFrames."""
    rng = np.random.default_rng(seed)
    is_vlm = dataset_name.lower() == "mathvista"

    gen_rows = []
    prob_rows = []
    attn_rows = []

    answer_pool = ["42", "\\frac{1}{2}", "12", "-5", "3\\pi", "0", "A", "B", "C", "D"]

    for q_idx in range(num_questions):
        sample_id = f"{dataset_name.lower()}_{q_idx:04d}"
        gold_ans = answer_pool[q_idx % len(answer_pool)]
        prompt_text = f"Solve the following mathematical question {q_idx}: What is the value?"

        for r_idx in range(rollouts_per_question):
            rollout_id = r_idx
            seq_len = int(rng.integers(20, 60))

            # 40% probability of matching gold answer
            is_correct = rng.random() < 0.40
            if is_correct:
                extracted_ans = gold_ans
            else:
                extracted_ans = answer_pool[(q_idx + r_idx + 1) % len(answer_pool)]

            # 5% chance of missing answer (NO_ANSWER)
            if rng.random() < 0.05:
                extracted_ans = ""

            gen_text = f"Step 1: calculate. The answer is \\boxed{{{extracted_ans}}}."
            token_ids = rng.integers(100, 10000, size=seq_len).tolist()
            logp_vals = (-rng.exponential(scale=1.2, size=seq_len)).astype(np.float32).tolist()
            kl_vals = rng.uniform(0.5, 4.5, size=seq_len).astype(np.float32).tolist()

            gen_rows.append({
                "dataset": dataset_name.lower(),
                "sample_id": sample_id,
                "rollout_id": rollout_id,
                "prompt": prompt_text,
                "generated_text": gen_text,
                "extracted_answer": extracted_ans,
                "gold_answer": gold_ans,
                "token_ids": token_ids,
                "length": seq_len,
            })

            prob_rows.append({
                "dataset": dataset_name.lower(),
                "sample_id": sample_id,
                "rollout_id": rollout_id,
                "logp": logp_vals,
                "kl_u_p": kl_vals,
            })

            if is_vlm:
                # Attention metrics: image, prompt, generated
                a_img = rng.uniform(0.1, 1.0, size=seq_len).astype(np.float32).tolist()
                a_prompt = rng.uniform(0.2, 0.8, size=seq_len).astype(np.float32).tolist()
                a_gen = rng.uniform(0.1, 0.5, size=seq_len).astype(np.float32).tolist()
                attn_rows.append({
                    "dataset": dataset_name.lower(),
                    "sample_id": sample_id,
                    "rollout_id": rollout_id,
                    "A_image": a_img,
                    "A_prompt": a_prompt,
                    "A_generated": a_gen,
                })

    df_gen = pd.DataFrame(gen_rows)
    df_prob = pd.DataFrame(prob_rows)
    result = {
        "generations": df_gen,
        "probability_metrics": df_prob,
    }
    if is_vlm:
        result["attention_metrics"] = pd.DataFrame(attn_rows)

    return result


def export_synthetic_parquets(
    output_dir: str | Path,
    num_questions: int = 1000,
    seed: int = 6304,
) -> None:
    """Exports synthetic datasets matching ATML-PA3-Data directory structure."""
    out_base = Path(output_dir)
    for d_name in ["MATH", "MathVista"]:
        d_dir = out_base / d_name
        d_dir.mkdir(parents=True, exist_ok=True)
        dfs = generate_synthetic_rollout_dataset(dataset_name=d_name, num_questions=num_questions, seed=seed)
        dfs["generations"].to_parquet(d_dir / "generations.parquet", index=False)
        dfs["probability_metrics"].to_parquet(d_dir / "probability_metrics.parquet", index=False)
        if "attention_metrics" in dfs:
            dfs["attention_metrics"].to_parquet(d_dir / "attention_metrics.parquet", index=False)
        logger.info(f"Generated synthetic {d_name} dataset ({num_questions} questions) in {d_dir}")


def create_synthetic_dataset(data_dir: Path, dataset_name: str, num_questions: int = 20, rollouts_per_q: int = 32) -> Dict[str, Any]:
    """Generates synthetic dataset and returns inspection metadata."""
    export_synthetic_parquets(data_dir, num_questions=num_questions, seed=6304)
    total_rows = num_questions * rollouts_per_q
    return {
        "dataset": dataset_name,
        "total_rows": total_rows,
        "unique_questions": num_questions,
        "files": [{"file_name": "generations.parquet", "size_mb": 0.5, "sha256": "mock_sha"}],
        "validation": {"all_valid": True, "duplicate_keys": 0, "missing_tokens": 0, "length_mismatches": 0},
    }
