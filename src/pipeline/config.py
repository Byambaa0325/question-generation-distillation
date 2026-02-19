"""PipelineConfig — loads config/pipeline.yaml into typed dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    raw:       str = "data/raw"
    processed: str = "data/processed"
    training:  str = "data/training"
    models:    str = "models"
    results:   str = "results"


@dataclass
class WikiConfig:
    tool:        str = "wikifier"
    top_n:       int = 5
    chunk_size:  int = 650
    save_every:  int = 50
    # Wikifier params
    df_ignore:    int = 200
    words_ignore: int = 200
    # WAT params
    wat_lang: str = "en"


@dataclass
class DatasetConfig:
    max_tokens:        int   = 510
    train_ratio:       float = 0.70
    val_ratio:         float = 0.15
    test_ratio:        float = 0.15
    mix_samples:       int   = 10000
    khanq_mix_samples: int   = 653
    seed:              int   = 42


@dataclass
class TrainingConfig:
    model_name:     str       = "google-t5/t5-small"
    batch:          int       = 64
    lr:             float     = 1e-3
    epochs:         int       = 50
    max_input_len:  int       = 200
    max_output_len: int       = 45
    warmup_steps:   int       = 0
    special_tokens: list[str] = field(default_factory=lambda: ["<sep>", "<space>"])


@dataclass
class EvalConfig:
    max_samples:           Optional[int] = None
    compute_wikisemrel:    bool          = False
    num_beams:             int           = 10
    num_return_sequences:  int           = 8
    zero_shot_models:      list[str]     = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Typed configuration loaded from ``config/pipeline.yaml``."""

    paths:        PathsConfig    = field(default_factory=PathsConfig)
    wikification: WikiConfig     = field(default_factory=WikiConfig)
    dataset:      DatasetConfig  = field(default_factory=DatasetConfig)
    training:     TrainingConfig = field(default_factory=TrainingConfig)
    evaluation:   EvalConfig     = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = "config/pipeline.yaml") -> "PipelineConfig":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()

        if "paths" in data:
            for k, v in data["paths"].items():
                if hasattr(cfg.paths, k):
                    setattr(cfg.paths, k, v)

        if "wikification" in data:
            for k, v in data["wikification"].items():
                if hasattr(cfg.wikification, k):
                    setattr(cfg.wikification, k, v)

        if "dataset" in data:
            for k, v in data["dataset"].items():
                if hasattr(cfg.dataset, k):
                    setattr(cfg.dataset, k, v)

        if "training" in data:
            tr = data["training"]
            for k, v in tr.items():
                if hasattr(cfg.training, k):
                    setattr(cfg.training, k, v)

        if "evaluation" in data:
            for k, v in data["evaluation"].items():
                if hasattr(cfg.evaluation, k):
                    setattr(cfg.evaluation, k, v)

        return cfg

    # ------------------------------------------------------------------
    # Resolved path helpers
    # ------------------------------------------------------------------

    def processed_dir(self, tool: Optional[str] = None) -> Path:
        """
        Return the processed-data directory.

        If *tool* is given, returns ``<processed>/<tool>/``.
        If not given, uses ``self.wikification.tool``.
        If the tool is ``None`` or empty, returns the bare processed dir.
        """
        base = Path(self.paths.processed)
        t = tool if tool is not None else self.wikification.tool
        return base / t if t else base

    def training_dir(self, dataset: str, mode: str) -> Path:
        """Return ``<training>/<dataset>/<mode>/``."""
        return Path(self.paths.training) / dataset / mode

    def model_dir(self, mode: str) -> Path:
        """Return ``<models>/<mode>/``."""
        return Path(self.paths.models) / mode

    def raw_path(self, filename: str) -> Path:
        return Path(self.paths.raw) / filename

    def get_wikifier_tool(self):
        """Construct and return the configured wikifier instance."""
        from src.wikification import get_wikifier

        return get_wikifier(self.wikification.tool, self)
