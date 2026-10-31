"""Configuration loading, validation, and dataclass structures for Task 2.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def load_yaml(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ConfigNamespace:
    """Dictionary wrapper enabling both attribute-style and dict-style access."""
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNamespace(v))
            else:
                setattr(self, k, v)
        self._raw = d

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class StageConfig:
    stage_id: int
    name: str
    updates: int
    connector_lr: float
    lora_lr: Optional[float]


class Task2Config:
    """Master configuration for Task 2."""

    def __init__(self, raw: Dict[str, Any], stages_raw: Dict[str, Any], eval_raw: Dict[str, Any], base_dir: Path):
        self.raw_config = raw
        self.base_dir = base_dir

        # 1. Models
        m = raw.get("models", {})
        v = m.get("vision_encoder", {})
        conn = m.get("connector", {})
        lm = m.get("language_model", {})
        lora = raw.get("lora", {})

        self.models = ConfigNamespace({
            "siglip_model_id": v.get("name_or_path", "google/siglip-base-patch16-224"),
            "qwen_model_id": lm.get("name_or_path", "Qwen/Qwen3-0.6B"),
            "vision_hidden_dim": v.get("hidden_dim", 768),
            "qwen_hidden_dim": conn.get("out_dim", 1024),
            "vision_revision": v.get("revision", "main"),
            "language_revision": lm.get("revision", "main"),
            "enable_thinking": lm.get("enable_thinking", False),
            "apply_final_ln_once": v.get("apply_final_ln_once", True),
            "lora_r": lora.get("r", 16),
            "lora_alpha": lora.get("alpha", 32),
            "lora_dropout": lora.get("dropout", 0.05),
        })

        # 2. Paths
        s = raw.get("storage", {})
        c = raw.get("checkpointing", {})

        p_root = Path(s.get("persistent_root", "/content/drive/MyDrive/ATML_PA3/task2"))
        f_cache = base_dir / s.get("feature_cache_dir", "feature_cache")
        ckpts = base_dir / c.get("save_dir", "checkpoints")
        res = base_dir / s.get("results_dir", "results")
        logs = base_dir / s.get("logs_dir", "logs")
        arts = base_dir / s.get("artifacts_dir", "artifacts")

        self.paths = ConfigNamespace({
            "drive_output_root": str(p_root),
            "feature_cache_dir": str(f_cache),
            "checkpoints_dir": str(ckpts),
            "results_dir": str(res),
            "logs_dir": str(logs),
            "artifacts_dir": str(arts),
        })

        # Top-level path aliases
        self.feature_cache_dir = f_cache
        self.checkpoints_dir = ckpts
        self.results_dir = res
        self.logs_dir = logs
        self.artifacts_dir = arts

        # 3. Data
        d = raw.get("data", {})
        self.data = ConfigNamespace({
            "seed": raw.get("seed", 6304),
            "train_images_count": d.get("coco_captions", {}).get("train_images", 6000),
            "captions_per_train_image": d.get("coco_captions", {}).get("captions_per_train_image", 3),
            "dev_images_count": d.get("coco_captions", {}).get("dev_images", 96),
        })
        self.seed = raw.get("seed", 6304)

        # 4. Training
        t = raw.get("training", {})
        caps = t.get("content_caps", {})
        self.training = ConfigNamespace({
            "total_updates": t.get("total_updates", 4500),
            "microbatch_size": t.get("microbatch_size", 8),
            "gradient_accumulation_steps": t.get("gradient_accumulation_steps", 2),
            "effective_batch_size": t.get("effective_batch_size", 16),
            "weight_decay": t.get("weight_decay", 0.01),
            "max_grad_norm": t.get("max_grad_norm", 1.0),
            "warmup_ratio": t.get("warmup_ratio", 0.05),
            "min_lr_ratio": t.get("min_lr_ratio", 0.10),
            "question_content_cap": caps.get("question", 80),
            "caption_content_cap": caps.get("caption_target", 64),
            "short_answer_content_cap": caps.get("short_answer_target", 16),
            "save_interval_updates": c.get("save_interval_updates", 250),
        })

        # 5. Stages
        self.stages: Dict[int, StageConfig] = {
            1: StageConfig(1, "Caption Alignment", 2250, 1e-3, None),
            2: StageConfig(2, "Visual Instruction Tuning", 1500, 1e-4, 2e-5),
            3: StageConfig(3, "Natural VQA Specialization", 750, 5e-5, 1e-5),
        }
        if "stages" in stages_raw:
            for s_info in stages_raw["stages"]:
                sid = s_info.get("stage")
                if sid in self.stages:
                    self.stages[sid].updates = s_info.get("updates", self.stages[sid].updates)
                    self.stages[sid].connector_lr = s_info.get("connector_lr", self.stages[sid].connector_lr)
                    self.stages[sid].lora_lr = s_info.get("lora_lr", self.stages[sid].lora_lr)

        # 6. Evaluation
        sched = eval_raw.get("development", {}).get("evaluation_schedule", [0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500])
        self.evaluation = ConfigNamespace({
            "dev_eval_schedule": sched,
            "greedy_max_new_tokens": 16,
            "bootstrap_resamples": 4000,
        })


def load_task2_config(config_path: Path | str) -> Task2Config:
    p = Path(config_path)
    base = p.parent.parent
    raw = load_yaml(p)
    stages_raw = load_yaml(p.parent / "stages.yaml")
    eval_raw = load_yaml(p.parent / "evaluation.yaml")
    return Task2Config(raw=raw, stages_raw=stages_raw, eval_raw=eval_raw, base_dir=base)
