"""Regression tests for activation probe extraction, manifest repair, and activation caching."""

import json
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn

from task1.probes import (
    _sanitize_and_repair_probes,
    cache_base_model_activations,
    extract_probe_inputs,
    sample_probe_indices,
)


class MockTokenizer:
    """Mock tokenizer simulating Qwen chat template and encoding."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 151645

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        # Return a list of integer token IDs
        return [151644, 872, 198, 100, 200, 151645]

    def encode(self, text, add_special_tokens=True):
        return [300, 400, 500]


class DummyMLP(nn.Module):
    def __init__(self, d_in=16, d_out=16):
        super().__init__()
        self.down_proj = nn.Linear(d_in, d_out)


class DummyLayer(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.mlp = DummyMLP(d_model, d_model)

    def forward(self, x):
        return self.mlp.down_proj(x)


class DummyTransformerModel(nn.Module):
    def __init__(self, num_layers=28, d_model=16):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(d_model) for _ in range(num_layers)])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size=1000, num_layers=28, d_model=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = DummyTransformerModel(num_layers=num_layers, d_model=d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)
        # Manually invoke down_proj forward so the registered hook triggers
        seq_len = x.size(1)
        for layer_idx in [0, 13, 27]:
            # Simulate down_proj input tensor of shape (batch, seq_len, d_in)
            _ = self.model.layers[layer_idx].mlp.down_proj(x)
        return self.lm_head(x)


class TestProbes(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_sample_probe_indices_determinism(self):
        idx1 = sample_probe_indices(seed=6304)
        idx2 = sample_probe_indices(seed=6304)
        self.assertEqual(idx1["gsm8k_indices"], idx2["gsm8k_indices"])
        self.assertEqual(idx1["hellaswag_indices"], idx2["hellaswag_indices"])
        self.assertEqual(len(idx1["gsm8k_indices"]), 32)
        self.assertEqual(len(idx1["hellaswag_indices"]), 32)

    def test_repair_stale_manifest_with_dict_keys(self):
        """
        Regression test for ValueError: too many dimensions 'str'.
        Simulates stale cached manifest where input_ids contained dict keys:
        ['input_ids', 'attention_mask'].
        """
        malformed_probes = []
        for i in range(32):
            malformed_probes.append({
                "probe_id": f"gsm8k_{i}",
                "group": "math",
                "raw_text": f"Solve problem {i}",
                "input_ids": ["input_ids", "attention_mask"],
                "seq_len": 2,
            })
        for i in range(32):
            malformed_probes.append({
                "probe_id": f"hellaswag_{i}",
                "group": "non_math",
                "raw_text": f"Context description {i}",
                "input_ids": ["input_ids", "attention_mask"],
                "seq_len": 2,
            })

        repaired, was_modified = _sanitize_and_repair_probes(malformed_probes, tokenizer=self.tokenizer)
        self.assertTrue(was_modified)
        self.assertEqual(len(repaired), 64)

        for p in repaired:
            # Must be integer token IDs
            self.assertTrue(all(isinstance(t, int) for t in p["input_ids"]))
            self.assertGreater(len(p["input_ids"]), 2)
            # torch.tensor conversion must succeed with torch.long
            tensor_ids = torch.tensor([p["input_ids"]], dtype=torch.long)
            self.assertEqual(tensor_ids.dtype, torch.long)

    def test_repair_numeric_strings(self):
        """Tests that numeric string tokens (e.g. ['101', '102']) are coerced to ints."""
        probes = [
            {
                "probe_id": "test_0",
                "group": "math",
                "raw_text": "2+2",
                "input_ids": ["101", "102", "103"],
                "seq_len": 3,
            }
        ]
        repaired, was_modified = _sanitize_and_repair_probes(probes, tokenizer=self.tokenizer)
        self.assertTrue(was_modified)
        self.assertEqual(repaired[0]["input_ids"], [101, 102, 103])
        self.assertTrue(all(isinstance(t, int) for t in repaired[0]["input_ids"]))

    def test_extract_probe_inputs_heals_corrupted_cached_file(self):
        """Tests that loading a corrupted cached manifest file repairs and overwrites it on disk."""
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            corrupted_data = {
                "seed": 6304,
                "math_count": 32,
                "non_math_count": 32,
                "total_probes": 64,
                "probes": [
                    {
                        "probe_id": f"probe_{i}",
                        "group": "math" if i < 32 else "non_math",
                        "raw_text": f"Sample question {i}",
                        "input_ids": ["input_ids", "attention_mask"],
                        "seq_len": 2,
                    }
                    for i in range(64)
                ],
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(corrupted_data, f, indent=2)

            # Load via extract_probe_inputs with tokenizer
            probes, manifest = extract_probe_inputs(self.tokenizer, seed=6304, cache_path=temp_path)
            self.assertEqual(len(probes), 64)
            for p in probes:
                self.assertTrue(all(isinstance(t, int) for t in p["input_ids"]))

            # Verify that the file on disk was healed
            with open(temp_path, "r", encoding="utf-8") as f:
                saved_manifest = json.load(f)
            first_probe = saved_manifest["probes"][0]
            self.assertTrue(all(isinstance(t, int) for t in first_probe["input_ids"]))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_cache_base_model_activations_execution(self):
        """Verifies cache_base_model_activations executes without dimension or string errors."""
        model = DummyCausalLM(vocab_size=1000, num_layers=28, d_model=16)
        probes = [
            {
                "probe_id": f"probe_{i}",
                "group": "math",
                "raw_text": f"Test {i}",
                "input_ids": [10, 20, 30 + i],
                "seq_len": 3,
            }
            for i in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = cache_base_model_activations(
                base_model=model,
                probes=probes,
                target_layers=[0, 13, 27],
                output_dir=tmpdir,
                device=torch.device("cpu"),
            )
            self.assertEqual(len(cache), 2 * 3)  # 2 probes * 3 layers = 6
            for key, val in cache.items():
                self.assertEqual(val.shape, torch.Size([16]))
                self.assertEqual(val.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

