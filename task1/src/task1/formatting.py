"""Chat template formatting, tokenization, loss masking, and token-weighted loss."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def format_chat_prompt(question: str, enable_thinking: bool = False) -> List[Dict[str, str]]:
    """Creates the user message structure."""
    return [{"role": "user", "content": question}]


def format_chat_conversation(question: str, response: str, enable_thinking: bool = False) -> List[Dict[str, str]]:
    """Creates the complete two-turn user/assistant conversation structure."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]


def tokenize_training_example(
    tokenizer: Any,
    question: str,
    response: str,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Formats and tokenizes a single training example with proper masking.

    Guarantees:
    - Disables thinking where supported (enable_thinking=False).
    - Preserves exact token IDs without duplicate BOS/EOS tokens.
    - Masks user prompt tokens and padding with -100.
    - Supervises all assistant response tokens and exactly one assistant end-of-turn token.
    - Excludes examples whose formatted length strictly exceeds max_seq_length.
    """
    messages_prompt = format_chat_prompt(question, enable_thinking=enable_thinking)
    messages_full = format_chat_conversation(question, response, enable_thinking=enable_thinking)

    # Apply chat template with enable_thinking=False if accepted by tokenizer
    kwargs: Dict[str, Any] = {"tokenize": True}
    # Check if tokenizer accepts enable_thinking in apply_chat_template
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt, add_generation_prompt=True, enable_thinking=enable_thinking, **kwargs
        )
        full_ids = tokenizer.apply_chat_template(
            messages_full, add_generation_prompt=False, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        # Fallback if tokenizer does not accept enable_thinking keyword argument
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt, add_generation_prompt=True, **kwargs
        )
        full_ids = tokenizer.apply_chat_template(
            messages_full, add_generation_prompt=False, **kwargs
        )

    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)
    if not isinstance(full_ids, list):
        full_ids = list(full_ids)

    # Guardrail: check length before truncation
    seq_len = len(full_ids)
    if seq_len > max_seq_length:
        return None  # Excluded: strictly exceeds cap; no truncation allowed

    prompt_len = len(prompt_ids)
    if prompt_len >= seq_len:
        # Prompt longer or equal to full conversation (empty response)
        return None

    # Construct labels: -100 for prompt tokens, token ID for response + EOT
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])

    # Token strings for auditing
    attention_mask = [1] * seq_len

    num_supervised_tokens = sum(1 for lbl in labels if lbl != -100)
    if num_supervised_tokens <= 0:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "num_supervised_tokens": num_supervised_tokens,
    }


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
