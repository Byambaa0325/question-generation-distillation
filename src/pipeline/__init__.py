"""
Importable Pipeline class for topic-controlled question generation.

Usage
-----
>>> from src.pipeline import Pipeline
>>> pipe = Pipeline('config/pipeline.yaml')
>>> pipe.convert(dataset='squad')
>>> pipe.wikify(dataset='squad', tool='wikifier')
>>> pipe.topics(dataset='squad')
>>> pipe.dataset(dataset='squad', mode='mixsquad')
>>> pipe.train(mode='topic')
>>> results = pipe.evaluate()
>>> q = pipe.generate(topic='Photosynthesis', context='Plants convert...')
>>> pipe.status()
>>> pipe.run(dataset='squad')
"""

from pathlib import Path
from typing import Optional

from src.pipeline.config import PipelineConfig

# Import stages as modules (heavy work deferred inside each run() function)
from src.pipeline.stages import convert as _convert_stage
from src.pipeline.stages import wikify as _wikify_stage
from src.pipeline.stages import topics as _topics_stage
from src.pipeline.stages import dataset as _dataset_stage
from src.pipeline.stages import train as _train_stage
from src.pipeline.stages import evaluate as _evaluate_stage
from src.pipeline.stages import generate as _generate_stage

__all__ = ["Pipeline"]


class Pipeline:
    """
    High-level interface to the topic-QG pipeline.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file (default: ``config/pipeline.yaml``).
    """

    def __init__(self, config_path: str | Path = "config/pipeline.yaml"):
        self.config = PipelineConfig.load(config_path)

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def convert(self, dataset: str = "all") -> dict[str, Path]:
        """Stage 1: Raw JSON → wikifier-ready flat JSON."""
        return _convert_stage.run(self.config, dataset=dataset)

    def wikify(
        self,
        dataset: str = "all",
        tool: Optional[str] = None,
        target: str = "all",
    ) -> dict[str, Path]:
        """Stage 2: Annotate texts and questions with Wikipedia entities."""
        return _wikify_stage.run(self.config, dataset=dataset, tool=tool, target=target)

    def topics(
        self,
        dataset: str = "all",
        tool: Optional[str] = None,
    ) -> dict[str, Path]:
        """Stage 3: Select best topic per QA pair (intersection + highest score)."""
        return _topics_stage.run(self.config, dataset=dataset, tool=tool)

    def dataset(
        self,
        dataset: str = "squad",
        mode: str = "mixsquad",
        tool: Optional[str] = None,
    ) -> dict[str, Path]:
        """
        Stage 4: Build train/val/test CSVs.

        mode ∈ {baseline, mixsquad, mixsquad2x, mixkhanq}
        """
        return _dataset_stage.run(self.config, dataset=dataset, mode=mode, tool=tool)

    def train(
        self,
        mode: str = "topic",
        dataset: str = "squad",
        tool: Optional[str] = None,
    ) -> Path:
        """
        Stage 5: Fine-tune T5-small.

        mode ∈ {baseline, topic, topic2x}
        """
        return _train_stage.run(self.config, mode=mode, dataset=dataset, tool=tool)

    def evaluate(
        self,
        models: str = "all",
        dataset: str = "all",
        output_dir: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> dict:
        """Stage 6: Evaluate all model variants with full metric suite."""
        return _evaluate_stage.run(
            self.config,
            models=models,
            dataset=dataset,
            output_dir=output_dir,
            tool=tool,
        )

    def generate(
        self,
        topic: str,
        context: str,
        model_path: Optional[str | Path] = None,
        mode: str = "topic",
        **kwargs,
    ) -> str:
        """Stage 7: Generate a question using beam search + reranking."""
        return _generate_stage.run(
            self.config,
            topic=topic,
            context=context,
            model_path=model_path,
            mode=mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def status(self) -> dict[str, bool]:
        """
        Check which pipeline outputs exist.

        Returns a dict mapping stage/file labels to booleans.
        """
        checks: dict[str, bool] = {}
        processed = Path(self.config.paths.processed)
        tool = self.config.wikification.tool
        wiki_dir = self.config.processed_dir(tool)

        # Stage 1 — convert
        for ds in ("squad", "khanq"):
            for kind in ("text", "question"):
                checks[f"convert.{ds}.{kind}"] = (
                    processed / f"ready_{ds}_{kind}.json"
                ).exists()

        # Stage 2 — wikify
        for ds in ("squad", "khanq"):
            for kind in ("text", "question"):
                checks[f"wikify.{ds}.{kind}"] = (
                    wiki_dir / f"wikified_{ds}_{kind}.json"
                ).exists()

        # Stage 3 — topics
        for ds in ("squad", "khanq"):
            checks[f"topics.{ds}.enriched"] = (wiki_dir / f"enriched_{ds}.json").exists()
            checks[f"topics.{ds}.filtered"] = (wiki_dir / f"enriched_{ds}_filtered.json").exists()

        # Stage 4 — dataset
        for ds in ("squad", "khanq"):
            for mode_label in ("baseline", "mixsquad", "mixsquad2x"):
                checks[f"dataset.{ds}.{mode_label}"] = (
                    self.config.training_dir(ds, mode_label) / "train.csv"
                ).exists()

        # Stage 5 — train
        for mode_label in ("baseline", "topic", "topic2x"):
            best = self.config.model_dir(mode_label) / "best_model"
            checks[f"train.{mode_label}"] = best.exists() and any(best.iterdir())

        self._print_status(checks)
        return checks

    @staticmethod
    def _print_status(checks: dict[str, bool]) -> None:
        print("\nPipeline status:")
        for key, exists in checks.items():
            marker = "+" if exists else "-"
            print(f"  [{marker}] {key}")

    def run(self, dataset: str = "squad", skip: Optional[list[str]] = None) -> None:
        """
        Run the full pipeline end-to-end.

        Parameters
        ----------
        dataset:
            ``"squad"``, ``"khanq"``, or ``"all"``.
        skip:
            List of stage names to skip, e.g. ``["wikify", "topics"]``.
        """
        skip_set = {s.lower() for s in (skip or [])}
        ds = dataset

        if "convert" not in skip_set:
            self.convert(dataset=ds)

        if "wikify" not in skip_set:
            self.wikify(dataset=ds)

        if "topics" not in skip_set:
            self.topics(dataset=ds)

        if "dataset" not in skip_set:
            train_ds = "squad" if ds == "all" else ds
            mode = "mixkhanq" if train_ds == "khanq" else "mixsquad"
            self.dataset(dataset=train_ds, mode=mode)

        if "train" not in skip_set and ds != "khanq":
            train_ds = "squad" if ds == "all" else ds
            self.train(mode="topic", dataset=train_ds)

        if "evaluate" not in skip_set:
            self.evaluate(dataset=ds)
