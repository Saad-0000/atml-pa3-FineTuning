"""
Live generation experiment runner comparing Single-Long, Parallel, and Sequential Refinement.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from task3.src.scoring.boxed_extractor import extract_last_boxed_answer
from task3.src.scoring.normalization import NO_ANSWER, grade_answer, normalize_math_answer
from task3.src.selection.selectors import select_self_consistency
from task3.src.utils.common import atomic_save_json, setup_logger

logger = setup_logger("task3.live_generation")


def format_initial_messages(problem: str) -> List[Dict[str, str]]:
    """Constructs prompt for initial generation."""
    prompt_content = f"{problem.strip()}\n\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{{}}."
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_content}
    ]


def format_refinement_messages(problem: str, initial_response: str) -> List[Dict[str, str]]:
    """Constructs prompt for sequential refinement reusing the initial response."""
    initial_prompt = f"{problem.strip()}\n\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{{}}."
    refinement_instruction = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": initial_prompt},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": refinement_instruction}
    ]


class LiveGenerationRunner:
    """Orchestrates live generation with Qwen2.5-3B-Instruct across 3 compute allocation strategies."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[torch.device] = None,
        output_dir: str | Path = "task3/results/live_generation",
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock

        self.model = None
        self.tokenizer = None
        if not self.use_mock:
            self._init_model()

    def _init_model(self):
        """Loads model and tokenizer in FP16 with no quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading {self.model_id} in FP16 onto {self.device}...")
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map={"": self.device},
                trust_remote_code=True,
            )
            self.model.eval()
            load_time = time.time() - t0
            logger.info(f"Loaded {self.model_id} in {load_time:.2f}s")
        except Exception as e:
            logger.warning(f"Could not load live model ({e}). Falling back to mock generator.")
            self.use_mock = True

    def generate_single_call(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        seed: int = 6304,
    ) -> Dict[str, Any]:
        """Executes a single generation call and returns raw text, tokens, duration, and peak memory."""
        t0 = time.time()
        if self.use_mock or self.model is None:
            # Deterministic synthetic mock generation
            gen_text = f"<think>Mock calculation for step</think> Final value is \\boxed{{42}}"
            duration = 0.05
            in_tokens = 50
            out_tokens = 25
            peak_mb = 0.0
        else:
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
            in_tokens = inputs.input_ids.shape[1]

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            torch.manual_seed(seed)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            duration = time.time() - t0
            out_ids = outputs[0][in_tokens:]
            out_tokens = len(out_ids)
            gen_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            peak_mb = (torch.cuda.max_memory_allocated(self.device) / (1024**2)) if self.device.type == "cuda" else 0.0

        boxed = extract_last_boxed_answer(gen_text)
        return {
            "raw_text": gen_text,
            "extracted_answer": boxed or NO_ANSWER,
            "normalized_answer": normalize_math_answer(boxed) if boxed else NO_ANSWER,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "duration": duration,
            "peak_memory_mb": peak_mb,
            "truncated": out_tokens >= max_new_tokens,
        }


class LiveGenerationPipeline:
    """Wrapper supporting config, dry_run, and question running."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str | Path = "task3/results/live_generation",
        dry_run: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runner = LiveGenerationRunner(output_dir=self.output_dir, use_mock=dry_run)

    def run_question(self, q_item: Dict[str, Any]) -> Dict[str, Any]:
        qid = q_item["question_id"]
        prob = q_item["problem"]
        gold = q_item.get("gold_answer", "")

        # 1. Strategy: Parallel Candidate 0 (cap 1024) -> Reused for Refinement!
        par_call_0 = self.runner.generate_single_call(format_initial_messages(prob), max_new_tokens=1024, seed=6304)
        
        # Candidate 1 and Candidate 2
        par_call_1 = self.runner.generate_single_call(format_initial_messages(prob), max_new_tokens=1024, seed=6305)
        par_call_2 = self.runner.generate_single_call(format_initial_messages(prob), max_new_tokens=1024, seed=6306)

        par_cands = [par_call_0, par_call_1, par_call_2]
        sc_res = select_self_consistency(par_cands, dataset="math")
        par_winner = sc_res["selected_answer"]
        par_correct = grade_answer(par_winner, gold, dataset="math")

        # 2. Strategy: Single Long (cap 3072)
        single_call = self.runner.generate_single_call(format_initial_messages(prob), max_new_tokens=3072, seed=6304)
        single_correct = grade_answer(single_call["normalized_answer"], gold, dataset="math")

        # 3. Strategy: Sequential Refinement (Reuse Candidate 0, then critique cap 2048)
        shared_init = par_call_0["raw_text"]
        ref_call = self.runner.generate_single_call(
            format_refinement_messages(prob, shared_init),
            max_new_tokens=2048,
            seed=6307
        )
        ref_correct = grade_answer(ref_call["normalized_answer"], gold, dataset="math")
        init_correct = grade_answer(par_call_0["normalized_answer"], gold, dataset="math")

        res = {
            "question_id": qid,
            "problem": prob,
            "gold_answer": gold,
            "strategies": {
                "single_long": {
                    "selected_answer": single_call["normalized_answer"],
                    "is_correct": single_correct,
                    "input_tokens": single_call["input_tokens"],
                    "output_tokens": single_call["output_tokens"],
                    "runtime_seconds": single_call["duration"],
                    "peak_memory_mb": single_call["peak_memory_mb"],
                    "extraction_success": single_call["extracted_answer"] != NO_ANSWER,
                    "truncated": single_call["truncated"],
                },
                "parallel": {
                    "candidates": par_cands,
                    "selected_answer": par_winner,
                    "is_correct": par_correct,
                    "input_tokens": sum(c["input_tokens"] for c in par_cands),
                    "output_tokens": sum(c["output_tokens"] for c in par_cands),
                    "runtime_seconds": sum(c["duration"] for c in par_cands),
                    "peak_memory_mb": max(c["peak_memory_mb"] for c in par_cands),
                    "extraction_success": par_winner != NO_ANSWER,
                    "truncated": any(c["truncated"] for c in par_cands),
                },
                "sequential_refinement": {
                    "initial_response": shared_init,
                    "revised_response": ref_call["raw_text"],
                    "selected_answer": ref_call["normalized_answer"],
                    "is_correct": ref_correct,
                    "initial_is_correct": init_correct,
                    "input_tokens": par_call_0["input_tokens"] + ref_call["input_tokens"],
                    "output_tokens": par_call_0["output_tokens"] + ref_call["output_tokens"],
                    "runtime_seconds": par_call_0["duration"] + ref_call["duration"],
                    "peak_memory_mb": max(par_call_0["peak_memory_mb"], ref_call["peak_memory_mb"]),
                    "extraction_success": ref_call["extracted_answer"] != NO_ANSWER,
                    "truncated": ref_call["truncated"],
                }
            }
        }
        return res

    def run_all_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        for q in questions:
            res = self.run_question(q)
            results[q["question_id"]] = res
        return results

    def summarize_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        n = max(len(all_results), 1)
        summary = {"strategies": {}, "refinement_transitions": {"wrong_to_correct": 0, "correct_to_wrong": 0, "correct_to_correct": 0, "wrong_to_wrong": 0}}

        for strat in ["single_long", "parallel", "sequential_refinement"]:
            acc = sum(1 for r in all_results.values() if r["strategies"][strat]["is_correct"]) / n
            in_tok = sum(r["strategies"][strat]["input_tokens"] for r in all_results.values()) / n
            out_tok = sum(r["strategies"][strat]["output_tokens"] for r in all_results.values()) / n
            rtime = sum(r["strategies"][strat]["runtime_seconds"] for r in all_results.values()) / n
            tot_rtime = sum(r["strategies"][strat]["runtime_seconds"] for r in all_results.values())
            mem = max((r["strategies"][strat]["peak_memory_mb"] for r in all_results.values()), default=0.0)

            summary["strategies"][strat] = {
                "accuracy": acc,
                "mean_input_tokens": in_tok,
                "mean_output_tokens": out_tok,
                "mean_runtime_sec": rtime,
                "total_runtime_sec": tot_rtime,
                "peak_memory_mb": mem,
                "num_questions": n,
            }

        # Transition matrix for refinement
        for r in all_results.values():
            s_ref = r["strategies"]["sequential_refinement"]
            init_c = s_ref["initial_is_correct"]
            fin_c = s_ref["is_correct"]
            if not init_c and fin_c:
                summary["refinement_transitions"]["wrong_to_correct"] += 1
            elif init_c and not fin_c:
                summary["refinement_transitions"]["correct_to_wrong"] += 1
            elif init_c and fin_c:
                summary["refinement_transitions"]["correct_to_correct"] += 1
            else:
                summary["refinement_transitions"]["wrong_to_wrong"] += 1

        return summary
