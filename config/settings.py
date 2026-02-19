"""Configuration management for the distillation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class OllamaConfig:
    model_name: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class GeminiConfig:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class TeacherConfig:
    backend: Literal["ollama", "gemini"] = "gemini"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


@dataclass
class StudentConfig:
    model_name: str = "google/flan-t5-base"
    max_input_length: int = 256
    max_output_length: int = 128


@dataclass
class ConceptConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    max_concepts_per_context: int = 5


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/distillation"
    num_epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 50
    eval_steps: int = 500
    early_stopping_patience: int = 3


@dataclass
class DataGenerationConfig:
    samples_per_concept_level: int = 3
    batch_size: int = 10
    output_dir: str = "data/generated"


@dataclass
class Settings:
    """Main settings container."""

    device: str = "cuda"
    seed: int = 42
    mixed_precision: bool = True

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    concepts: ConceptConfig = field(default_factory=ConceptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)

    learner_levels: list = field(default_factory=lambda: ["beginner", "intermediate", "advanced"])
    question_types: list = field(default_factory=lambda: ["recall", "explain", "apply"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        settings = cls()

        # Environment
        env = data.get("environment", {})
        settings.device = env.get("device", settings.device)
        settings.seed = env.get("seed", settings.seed)
        settings.mixed_precision = env.get("mixed_precision", settings.mixed_precision)

        # Teacher
        teacher_data = data.get("teacher", {})
        settings.teacher.backend = teacher_data.get("backend", settings.teacher.backend)
        if "ollama" in teacher_data:
            for k, v in teacher_data["ollama"].items():
                setattr(settings.teacher.ollama, k, v)
        if "gemini" in teacher_data:
            for k, v in teacher_data["gemini"].items():
                setattr(settings.teacher.gemini, k, v)

        # Student
        student_data = data.get("student", {})
        for k, v in student_data.items():
            setattr(settings.student, k, v)

        # Concepts
        concept_data = data.get("concepts", {})
        for k, v in concept_data.items():
            setattr(settings.concepts, k, v)

        # Training
        training_data = data.get("training", {})
        for k, v in training_data.items():
            setattr(settings.training, k, v)

        # Data generation
        datagen_data = data.get("data_generation", {})
        for k, v in datagen_data.items():
            setattr(settings.data_generation, k, v)

        # Levels and types
        if "learner_levels" in data:
            settings.learner_levels = [lvl["id"] for lvl in data["learner_levels"]]
        if "question_types" in data:
            settings.question_types = data["question_types"]

        return settings


_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        if config_path:
            _settings = Settings.from_yaml(config_path)
        else:
            default_path = Path(__file__).parent / "config.yaml"
            if default_path.exists():
                _settings = Settings.from_yaml(default_path)
            else:
                _settings = Settings()
    return _settings
