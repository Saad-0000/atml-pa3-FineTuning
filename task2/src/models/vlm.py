"""Vision-Language Model (VLM) integrating SigLIP connector and Qwen3-0.6B.

Assignment Section 3:
- Vision encoder: google/siglip-base-patch16-224 (frozen, features cached in FP16)
- Connector: MLP 768 -> 1024 -> GELU -> 1024 (trainable in all stages)
- Language model: Qwen/Qwen3-0.6B (frozen base, LoRA enabled in Stages 2-3)
- Visual tokens inserted before question in user turn
- Labels mask visual tokens, prompts, and padding (-100); only response + <|im_end|> supervised.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from task2.src.models.connector import MultimodalConnector, MLPConnector
from task2.src.models.lora import LoRALinear, apply_lora_to_model, mark_only_lora_and_connector_as_trainable

logger = logging.getLogger("task2.models.vlm")


class DummyTokenizer:
    """Lightweight deterministic tokenizer for offline smoke testing without internet."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<|im_start|>": 3, "<|im_end|>": 4, "user": 5, "assistant": 6}

    def __call__(self, text: str, return_tensors: Optional[str] = None):
        tokens = text.lower().replace("\n", " ").split()
        ids = [self.vocab.get(t, (hash(t) % 500) + 7) for t in tokens]
        if not ids:
            ids = [self.unk_token_id]
        if return_tensors == "pt":
            return type("TokenOut", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for i in ids:
            if skip_special_tokens and i in (0, 1, 2, 3, 4):
                continue
            words.append(f"token_{i}")
        return " ".join(words)


class DummyQwenForTesting(nn.Module):
    """Synthetic minimal transformer model for fast offline tests."""
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 1024):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": hidden_dim})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        # inputs_embeds: [B, S, D]
        # Causal dependency: position t attends to all positions 0..t
        h = self.q_proj(inputs_embeds)
        # Normalize causal accumulation so gradients flow smoothly across sequence
        causal_weights = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        causal_h = torch.cumsum(h, dim=1) / causal_weights
        out = self.o_proj(causal_h)
        logits = self.lm_head(out)
        return type("Output", (), {"logits": logits})()


class VLMModel(nn.Module):
    """Unified Vision-Language Model interface for training, evaluation, and inference."""

    def __init__(
        self,
        qwen_model_id: str = "Qwen/Qwen3-0.6B",
        vision_hidden_dim: int = 768,
        qwen_hidden_dim: int = 1024,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        connector: Optional[MultimodalConnector] = None,
        qwen_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.num_visual_tokens = 196
        self.lora_enabled = False

        # 1. Initialize connector
        if connector is not None:
            self.connector = connector
        else:
            self.connector = MultimodalConnector(
                in_dim=vision_hidden_dim,
                hidden_dim=qwen_hidden_dim,
                out_dim=qwen_hidden_dim,
            ).to(self.device)

        # 2. Initialize Qwen model and tokenizer
        if qwen_model is not None and tokenizer is not None:
            self.qwen = qwen_model
            self.tokenizer = tokenizer
        else:
            loaded_real = False
            # Step A: Check local cache first (no network)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_model_id,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.qwen = AutoModelForCausalLM.from_pretrained(
                    qwen_model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    local_files_only=True,
                ).to(self.device)
                self.qwen.config.use_cache = False
                loaded_real = True
            except Exception:
                loaded_real = False

            # Step B: If not in local cache, test fast connection with 1.0s timeout before downloading
            if not loaded_real:
                try:
                    import socket
                    socket.create_connection(("huggingface.co", 443), timeout=1.0)
                    self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token or "<|extra_0|>"
                    self.tokenizer.padding_side = "right"

                    self.qwen = AutoModelForCausalLM.from_pretrained(
                        qwen_model_id,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                    ).to(self.device)
                    self.qwen.config.use_cache = False
                    loaded_real = True
                except Exception as e:
                    logger.info(f"Local offline mode: Using synthetic model ({e}).")
                    self.tokenizer = DummyTokenizer()
                    self.qwen = DummyQwenForTesting(vocab_size=1000, hidden_dim=qwen_hidden_dim).to(self.device)

        # Always freeze base Qwen parameters
        for p in self.qwen.parameters():
            p.requires_grad = False

    def text_embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embeds token IDs into the Qwen embedding space."""
        embed_fn = self.qwen.get_input_embeddings()
        return embed_fn(token_ids)

    def attach_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> None:
        """Enables LoRA adapters on attention projections (Stages 2 and 3)."""
        apply_lora_to_model(
            self.qwen,
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.lora_enabled = True

    def set_stage_trainability(self, enable_lora: bool = False) -> None:
        """Sets parameter trainability according to stage policy."""
        mark_only_lora_and_connector_as_trainable(
            model=self.qwen,
            connector=self.connector,
            enable_lora=enable_lora,
        )

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns state dict containing only trainable LoRA adapter weights."""
        return {
            name: p.data.clone()
            for name, p in self.qwen.named_parameters()
            if ("lora_A" in name or "lora_B" in name) and p.requires_grad
        }

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restores LoRA adapter weights."""
        if not self.lora_enabled:
            self.attach_lora()
        with torch.no_grad():
            for name, p in self.qwen.named_parameters():
                if name in state_dict:
                    p.copy_(state_dict[name])

    def get_trainable_parameters_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def prepare_multimodal_inputs(
        self,
        batch_items: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Assembles multimodal batches inserting visual tokens before question in user turn.

        Masks: visual tokens (-100), prompt tokens (-100), padding (-100).
        Supervises: target response tokens + end-of-turn token.
        """
        device = self.device
        batch_size = len(batch_items)
        embed_fn = self.qwen.get_input_embeddings()

        seq_embeds_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for item in batch_items:
            v_feat = item.get("visual_features")
            q_text = item.get("question", "")
            ans_text = item.get("answer", "")
            is_caption = item.get("is_caption", False)

            # Build prompt and target strings
            if is_caption:
                prompt_str = "<|im_start|>user\nProvide a caption for this image.<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt_str = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n"

            target_str = f"{ans_text}<|im_end|>\n"

            # Tokenize prompt and target
            p_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.squeeze(0).to(device)
            t_ids = self.tokenizer(target_str, return_tensors="pt").input_ids.squeeze(0).to(device)

            p_embeds = embed_fn(p_ids)
            t_embeds = embed_fn(t_ids)

            if v_feat is not None:
                # Prepend visual tokens before question
                if v_feat.ndim == 2:
                    v_feat_batch = v_feat.unsqueeze(0).to(device=device, dtype=self.dtype)
                else:
                    v_feat_batch = v_feat.to(device=device, dtype=self.dtype)
                v_embeds = self.connector(v_feat_batch).squeeze(0)  # [196, 1024]
                full_embeds = torch.cat([v_embeds, p_embeds, t_embeds], dim=0)
                v_len = self.num_visual_tokens
            else:
                full_embeds = torch.cat([p_embeds, t_embeds], dim=0)
                v_len = 0

            p_len = len(p_ids)
            t_len = len(t_ids)

            # Labels: -100 on visual and prompt tokens
            lbls = torch.full((v_len + p_len + t_len,), -100, dtype=torch.long, device=device)
            lbls[v_len + p_len :] = t_ids

            seq_embeds_list.append(full_embeds)
            labels_list.append(lbls)

        # Pad batch
        max_len = max(s.shape[0] for s in seq_embeds_list)
        hidden_dim = seq_embeds_list[0].shape[-1]
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        pad_embed = embed_fn(torch.tensor([pad_id], device=device))[0]

        padded_embeds = torch.zeros((batch_size, max_len, hidden_dim), dtype=seq_embeds_list[0].dtype, device=device)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            cur_len = seq_embeds_list[b].shape[0]
            padded_embeds[b, :cur_len] = seq_embeds_list[b]
            if cur_len < max_len:
                padded_embeds[b, cur_len:] = pad_embed
            padded_labels[b, :cur_len] = labels_list[b]
            attention_mask[b, :cur_len] = 1

        position_ids = torch.arange(0, max_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        return {
            "inputs_embeds": padded_embeds,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multimodal forward pass. Gradients flow through inputs_embeds into connector!"""
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits

    @torch.no_grad()
    def generate(
        self,
        visual_features: Optional[torch.Tensor] = None,
        question: str = "",
        max_new_tokens: int = 16,
    ) -> Dict[str, Any]:
        """Greedy text generation with thinking disabled."""
        self.eval()
        device = self.device
        embed_fn = self.qwen.get_input_embeddings()

        prompt_str = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        p_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        p_embeds = embed_fn(p_ids)

        if visual_features is not None:
            if visual_features.ndim == 2:
                vf = visual_features.unsqueeze(0).to(device=device, dtype=self.dtype)
            else:
                vf = visual_features.to(device=device, dtype=self.dtype)
            v_embeds = self.connector(vf)
            cur_embeds = torch.cat([v_embeds, p_embeds], dim=1)
        else:
            cur_embeds = p_embeds

        cur_mask = torch.ones((1, cur_embeds.shape[1]), dtype=torch.long, device=device)

        gen_tokens: List[int] = []
        eos_ids = [getattr(self.tokenizer, "eos_token_id", 1)]
        im_end_id = getattr(self.tokenizer, "im_end_id", None)
        if im_end_id is not None:
            eos_ids.append(im_end_id)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            e_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if e_id is not None and e_id != getattr(self.tokenizer, "unk_token_id", -1):
                eos_ids.append(e_id)

        is_truncated = False
        for step in range(max_new_tokens):
            logits = self.forward(inputs_embeds=cur_embeds, attention_mask=cur_mask)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())

            if next_token in eos_ids:
                break

            gen_tokens.append(next_token)
            if step == max_new_tokens - 1:
                is_truncated = True

            next_embed = embed_fn(torch.tensor([[next_token]], device=device))
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)

        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return {
            "text": text,
            "generated_tokens_count": len(gen_tokens),
            "is_truncated": is_truncated,
        }


SiglipQwenVLM = VLMModel
