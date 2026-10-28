"""Chat template formatting, tokenization, loss masking, and token-weighted loss."""

from typing import Any, Dict, List, Optional, Tuple
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("task1.formatting")


def format_chat_prompt(question: str, enable_thinking: bool = False) -> List[Dict[str, str]]:
    """Creates the user message structure."""
    return [{"role": "user", "content": question}]


def format_chat_conversation(question: str, response: str, enable_thinking: bool = False) -> List[Dict[str, str]]:
    """Creates the complete two-turn user/assistant conversation structure."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]


def _extract_token_ids(output: Any) -> List[int]:
    """
    Safely extracts a flat list of integer token IDs from tokenizer outputs.
    Handles:
    - Raw list of ints: [1, 2, 3]
    - BatchEncoding or dict with 'input_ids': {'input_ids': [1, 2, 3]}
    - PyTorch tensors / numpy arrays: tensor([1, 2, 3])
    - Nested batched lists: [[1, 2, 3]]
    """
    if output is None:
        return []
    if isinstance(output, dict) or hasattr(output, "data"):
        if "input_ids" in output:
            output = output["input_ids"]
    if hasattr(output, "tolist"):
        output = output.tolist()
    if not isinstance(output, list):
        try:
            output = list(output)
        except Exception:
            output = [output]
    if output and isinstance(output[0], list):
        output = output[0]
    return [int(t) for t in output]


def tokenize_training_example(
    tokenizer: Any,
    question: str,
    response: str,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
    return_diagnostics: bool = False,
) -> Any:
    """
    Formats and tokenizes a single training example with proper masking.

    Guarantees:
    - Disables thinking where supported (enable_thinking=False).
    - Preserves exact token IDs without duplicate BOS/EOS tokens.
    - Masks user prompt tokens and padding with -100.
    - Supervises all assistant response tokens and exactly one assistant end-of-turn token.
    - Excludes examples whose formatted length strictly exceeds max_seq_length.

    Args:
        tokenizer: Pre-trained tokenizer.
        question: User query string.
        response: Assistant target response string.
        max_seq_length: Maximum allowed total sequence length (2048 tokens).
        enable_thinking: Whether thinking mode is enabled (False for SFT).
        return_diagnostics: If True, returns (result_dict, reason_str, detail_dict).
                           If False, returns Optional[Dict[str, Any]].
    """
    messages_prompt = format_chat_prompt(question, enable_thinking=enable_thinking)
    messages_full = format_chat_conversation(question, response, enable_thinking=enable_thinking)

    kwargs: Dict[str, Any] = {"tokenize": True, "return_dict": False}
    prompt_raw = None
    full_raw = None

    try:
        prompt_raw = tokenizer.apply_chat_template(
            messages_prompt, add_generation_prompt=True, enable_thinking=enable_thinking, **kwargs
        )
        full_raw = tokenizer.apply_chat_template(
            messages_full, add_generation_prompt=False, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        # Fallback if tokenizer does not accept enable_thinking keyword argument
        try:
            prompt_raw = tokenizer.apply_chat_template(
                messages_prompt, add_generation_prompt=True, **kwargs
            )
            full_raw = tokenizer.apply_chat_template(
                messages_full, add_generation_prompt=False, **kwargs
            )
        except Exception as e:
            if return_diagnostics:
                return None, "tokenization_exception", {"exception": str(e)}
            return None
    except Exception as e:
        # Fallback if jinja evaluation raises exception with enable_thinking
        try:
            prompt_raw = tokenizer.apply_chat_template(
                messages_prompt, add_generation_prompt=True, **kwargs
            )
            full_raw = tokenizer.apply_chat_template(
                messages_full, add_generation_prompt=False, **kwargs
            )
        except Exception as e2:
            if return_diagnostics:
                return None, "tokenization_exception", {"exception": f"{e}; fallback: {e2}"}
            return None

    prompt_ids = _extract_token_ids(prompt_raw)
    full_ids = _extract_token_ids(full_raw)

    seq_len = len(full_ids)
    prompt_len = len(prompt_ids)

    # 1. Guardrail: check length before truncation
    if seq_len > max_seq_length:
        logger.debug(f"Excluded: seq_len {seq_len} > max {max_seq_length}")
        if return_diagnostics:
            return None, "overlength", {"seq_len": seq_len, "prompt_len": prompt_len, "max_seq_length": max_seq_length}
        return None

    # 2. Guardrail: prompt length must be strictly smaller than full sequence
    if prompt_len >= seq_len:
        logger.debug(f"Excluded: prompt_len {prompt_len} >= seq_len {seq_len}")
        if return_diagnostics:
            return None, "prompt_ge_seq", {"seq_len": seq_len, "prompt_len": prompt_len}
        return None

    # 3. Handle think tags boundary alignment if prompt template omitted think tokens
    # Empty think tag sequence in Qwen3 is [151667, 271, 151668, 271] (<think>\n\n</think>\n\n)
    empty_think_tokens = [151667, 271, 151668, 271]
    if (
        full_ids[:prompt_len] == prompt_ids
        and seq_len >= prompt_len + len(empty_think_tokens)
        and full_ids[prompt_len : prompt_len + len(empty_think_tokens)] == empty_think_tokens
        and not enable_thinking
    ):
        # Empty think tokens belong to header; mask them out as prompt tokens
        prompt_len += len(empty_think_tokens)

    # Construct labels: -100 for prompt tokens, token ID for response + EOT
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])
    attention_mask = [1] * seq_len

    num_supervised_tokens = sum(1 for lbl in labels if lbl != -100)
    if num_supervised_tokens <= 0:
        logger.debug("Excluded: num_supervised_tokens <= 0")
        if return_diagnostics:
            return None, "no_supervised_tokens", {"seq_len": seq_len, "prompt_len": prompt_len, "num_supervised": 0}
        return None

    res = {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "num_supervised_tokens": num_supervised_tokens,
    }
    if return_diagnostics:
        return res, "ok", {"seq_len": seq_len, "prompt_len": prompt_len, "num_supervised": num_supervised_tokens}
    return res


def collate_effective_batch(
    batch_examples: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Pads and collates a list of tokenized examples into batch tensors.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch_examples)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for ex in batch_examples:
        pad_len = max_len - len(ex["input_ids"])
        input_ids = ex["input_ids"] + [pad_token_id] * pad_len
        attention_mask = ex["attention_mask"] + [0] * pad_len
        labels = ex["labels"] + [-100] * pad_len

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def compute_token_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Computes causal language modeling loss summed over all valid response tokens.

    Returns:
        total_token_loss: Tensor scalar (sum of cross-entropies for valid tokens)
        num_valid_tokens: int count of supervised response tokens
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    valid_mask = shift_labels != -100
    num_valid_tokens = int(valid_mask.sum().item())

    if num_valid_tokens == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0

    loss_sum = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, num_valid_tokens

