"""Unit tests for models: connector, LoRA, and multimodal sequence construction.
"""
import unittest
import torch
import torch.nn as nn

from task2.src.models.connector import MLPConnector, calibrate_connector_scale
from task2.src.models.lora import LoRALinear, mark_only_lora_and_connector_as_trainable
from task2.src.models.vlm import VLMModel
from task2.src.training.loss import compute_example_weighted_loss


class TestModels(unittest.TestCase):
    def test_connector_forward_and_scaling(self):
        """Validates MLP connector forward output shape and scale calibration buffer."""
        connector = MLPConnector(input_dim=768, hidden_dim=1024, output_dim=1024)
        x = torch.randn(2, 196, 768)
        out = connector(x)

        self.assertEqual(out.shape, (2, 196, 1024))
        self.assertEqual(connector.scale.item(), 1.0)

        # Test scale adjustment
        connector.set_scale(1.5)
        out_scaled = connector(x)
        self.assertAlmostEqual(out_scaled.norm().item(), out.norm().item() * 1.5, places=2)

    def test_lora_linear_forward_backward(self):
        """Validates LoRA linear adaptation, freezing of base weight, and adapter gradient flow."""
        base_linear = nn.Linear(64, 64, bias=False)
        lora_mod = LoRALinear(base_linear, r=8, lora_alpha=16.0, lora_dropout=0.0)

        # Base weight must be frozen
        self.assertFalse(lora_mod.base_layer.weight.requires_grad)
        # LoRA matrices must be trainable
        self.assertTrue(lora_mod.lora_A.requires_grad)
        self.assertTrue(lora_mod.lora_B.requires_grad)

        x = torch.randn(4, 64)
        out = lora_mod(x)
        self.assertEqual(out.shape, (4, 64))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(lora_mod.lora_A.grad)
        self.assertIsNotNone(lora_mod.lora_B.grad)
        self.assertIsNone(lora_mod.base_layer.weight.grad)

    def test_multimodal_token_layout_and_label_masking(self):
        """Validates multimodal token ordering (196 visual tokens before question) and label masking."""
        device = torch.device("cpu")
        vlm = VLMModel(device=device)

        sample_batch = [
            {
                "visual_features": torch.randn(196, 768),
                "question": "What animal is this?",
                "answer": "dog",
                "is_caption": False,
            }
        ]

        prepared = vlm.prepare_multimodal_inputs(sample_batch)
        inputs_embeds = prepared["inputs_embeds"]
        labels = prepared["labels"]
        attention_mask = prepared["attention_mask"]

        # Visual tokens (196) must be present in the embedding sequence
        self.assertGreaterEqual(inputs_embeds.shape[1], 196)
        # Visual token positions must have label == -100
        self.assertTrue((labels[0, :196] == -100).all())
        # Attention mask must be 1 across all non-padding positions
        self.assertTrue((attention_mask[0, :196] == 1).all())
        # At least some target response tokens must have label != -100
        self.assertTrue((labels[0] != -100).any())

    def test_autograd_gradient_flow_through_frozen_backbone(self):
        """Proves connector receives nonzero gradients while Qwen base remains frozen."""
        device = torch.device("cpu")
        vlm = VLMModel(device=device)
        vlm.set_stage_trainability(enable_lora=False)

        sample_batch = [
            {
                "visual_features": torch.randn(196, 768),
                "question": "Is this a bicycle?",
                "answer": "yes",
                "is_caption": False,
            }
        ]

        prepared = vlm.prepare_multimodal_inputs(sample_batch)
        vlm.connector.zero_grad()

        logits = vlm(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
        )
        loss, _ = compute_example_weighted_loss(logits, prepared["labels"])
        loss.backward()

        # Connector parameters have nonzero grad
        for name, p in vlm.connector.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertFalse(torch.isnan(p.grad).any())
                self.assertGreater(p.grad.norm().item(), 0.0)

        # Qwen base parameters must be frozen
        for name, p in vlm.qwen.named_parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
