"""Configuration management for Task 1."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Task1Config:
    seed: int = 6304
    model_name: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "main"
    enable_thinking: bool = False
    torch_dtype: str = "float16"
    use_cache: bool = False

    budget_examples: int = 40000
    total_optimizer_updates: int = 1250
    microbatch_size: int = 2
    gradient_accumulation_steps: int = 16
    effective_batch_size: int = 32
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_updates: int = 38
    lr_scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    source_dataset: str = "meta-math/MetaMathQA"
    leakage_eval_dataset: str = "gsm8k"
    leakage_eval_split: str = "test"

    milestones_updates: List[int] = field(default_factory=lambda: [312, 625, 937, 1250])
    milestones_examples: List[int] = field(default_factory=lambda: [9984, 20000, 29984, 40000])
    emergency_save_interval_updates: int = 50
    keep_latest: int = 2

    persistent_root: Path = Path("/content/drive/MyDrive/ATML_PA3_Task1")
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")
    checkpoints_dir: Path = Path("checkpoints")
    logs_dir: Path = Path("logs")

    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str, base_dir: Optional[Path] = None) -> "Task1Config":
        raw = load_yaml(yaml_path)
        base = base_dir or Path(yaml_path).parent.parent

        m = raw.get("model", {})
        t = raw.get("training", {})
        d = raw.get("data", {})
        c = raw.get("checkpointing", {})
        s = raw.get("storage", {})

        artifacts = Path(s.get("artifacts_dir", "artifacts"))
        results = Path(s.get("results_dir", "results"))
        checkpoints = Path(c.get("save_dir", "checkpoints"))
        logs = Path(s.get("logs_dir", "logs"))

        if not artifacts.is_absolute():
            artifacts = base / artifacts
        if not results.is_absolute():
            results = base / results
        if not checkpoints.is_absolute():
            checkpoints = base / checkpoints
        if not logs.is_absolute():
            logs = base / logs

        return cls(
            seed=raw.get("seed", 6304),
            model_name=m.get("name_or_path", "Qwen/Qwen3-0.6B"),
            model_revision=m.get("revision", "main"),
            enable_thinking=m.get("enable_thinking", False),
            torch_dtype=m.get("torch_dtype", "float16"),
            use_cache=m.get("use_cache", False),
            budget_examples=t.get("budget_examples", 40000),
            total_optimizer_updates=t.get("total_optimizer_updates", 1250),
            microbatch_size=t.get("microbatch_size", 2),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 16),
            effective_batch_size=t.get("effective_batch_size", 32),
            max_seq_length=t.get("max_seq_length", 2048),
            gradient_checkpointing=t.get("gradient_checkpointing", True),
            weight_decay=t.get("weight_decay", 0.01),
            max_grad_norm=t.get("max_grad_norm", 1.0),
            warmup_updates=t.get("warmup_updates", 38),
            lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
            adam_beta1=t.get("adam_beta1", 0.9),
            adam_beta2=t.get("adam_beta2", 0.999),
            adam_epsilon=float(t.get("adam_epsilon", 1.0e-8)),
            source_dataset=d.get("source_dataset", "meta-math/MetaMathQA"),
            leakage_eval_dataset=d.get("leakage_eval_dataset", "gsm8k"),
            leakage_eval_split=d.get("leakage_eval_split", "test"),
            milestones_updates=c.get("milestones_updates", [312, 625, 937, 1250]),
            milestones_examples=c.get("milestones_examples", [9984, 20000, 29984, 40000]),
            emergency_save_interval_updates=c.get("emergency_save_interval_updates", 50),
            keep_latest=c.get("keep_latest", 2),
            persistent_root=Path(s.get("persistent_root", "/content/drive/MyDrive/ATML_PA3_Task1")),
            artifacts_dir=artifacts,
            results_dir=results,
            checkpoints_dir=checkpoints,
            logs_dir=logs,
            raw_config=raw,
        )


def get_default_configs(task1_root: Optional[Path] = None) -> tuple[Task1Config, Dict[str, Any], Dict[str, Any]]:
    if task1_root is None:
        # Resolves to repo/task1
        task1_root = Path(__file__).resolve().parent.parent.parent
    config_dir = task1_root / "configs"
    t1_cfg = Task1Config.from_yaml(config_dir / "task1.yaml", base_dir=task1_root)
    methods_cfg = load_yaml(config_dir / "methods.yaml")
    eval_cfg = load_yaml(config_dir / "evaluation.yaml")
    return t1_cfg, methods_cfg, eval_cfg
