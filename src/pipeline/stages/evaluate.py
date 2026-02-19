"""
Stage 6 — Evaluate model variants with the full paper metric suite.

Metrics
-------
BLEU-1/2/3/4 (word-level, correct method)
BLEU-1/2/3/4 (char-level, paper's original method — for direct comparison)
Token F1
METEOR
ROUGE-L
Perplexity (T5 fine-tuned models only)

Model keys
----------
``t5:baseline``, ``t5:topic``, ``t5:topic2x``   — fine-tuned T5 checkpoints
``ollama:<model_name>``                           — local Ollama (must be running)
``gemini:<model_name>``                           — Google Gemini (needs API key)

Zero-shot models can be configured in ``config/pipeline.yaml``::

    evaluation:
      zero_shot_models:
        - ollama:llama3.1:8b
        - gemini:gemini-2.0-flash-exp

Results are saved to ``results/evaluation_{timestamp}/evaluation.json``.

Notes
-----
* KhanQ uses ``mixkhanq/data.csv`` (no train/val split).
  ``models='all'`` auto-discovers available T5 checkpoints plus any
  ``zero_shot_models`` listed in the config.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline.config import PipelineConfig
from src.evaluation.ngram import bleu_ngrams, bleu_ngrams_char
from src.evaluation.generation import compute_rouge_l, compute_meteor, compute_perplexity
from src.evaluation.models import load_model


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(
    config: PipelineConfig,
    models: str = "all",
    dataset: str = "all",
    output_dir: Optional[str] = None,
    tool: Optional[str] = None,
) -> dict:
    """
    Evaluate model variants and return a results dict.

    Parameters
    ----------
    models:
        ``"all"`` (auto-discovers T5 + config zero_shot_models) or a
        comma-separated list such as
        ``"t5:baseline,t5:topic,ollama:llama3.1:8b,gemini:gemini-2.0-flash-exp"``.
    dataset:
        ``"squad"``, ``"khanq"``, or ``"all"``.
    output_dir:
        Override output directory (default: ``results/evaluation_{ts}/``).

    Returns
    -------
    Nested dict: ``{"{dataset}/{model_key}": {metric: value, ...}}``.
    """
    import nltk
    for resource in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (
        Path(output_dir)
        if output_dir
        else Path(config.paths.results) / f"evaluation_{timestamp}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] evaluate -> {exp_dir}")

    effective_tool = (tool or config.wikification.tool).lower()
    all_results: dict = {}

    for ds in _datasets(dataset):
        test_csv = _find_test_csv(config, ds, effective_tool)
        if test_csv is None:
            print(f"  [SKIP] no test CSV found for dataset={ds}")
            continue

        test_data = pd.read_csv(test_csv)
        max_s = config.evaluation.max_samples
        if max_s:
            test_data = test_data.head(max_s)
        print(f"  Dataset: {ds} — {len(test_data)} examples  (from {test_csv.name})")

        for model_key in _resolve_models(config, models, ds):
            try:
                model_obj = load_model(config, model_key)
                metrics = _evaluate_model(model_obj, test_data, model_key, dataset=ds)
                all_results[f"{ds}/{model_key}"] = metrics
            except Exception as exc:
                print(f"  [ERROR] {model_key}: {exc}")

    # Save (strip non-serialisable fields)
    out_path = exp_dir / "evaluation.json"
    saveable = {
        k: {mk: mv for mk, mv in v.items() if mk not in ("predictions", "references")}
        for k, v in all_results.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Results saved to {out_path}")

    _print_table(all_results, dataset=dataset)
    return all_results


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def _evaluate_model(
    model_obj, test_data: pd.DataFrame, model_name: str, dataset: str = "squad"
) -> dict:
    bleu_w = {f"bleu{n}": [] for n in range(1, 5)}       # word-level
    bleu_c = {f"bleu{n}_char": [] for n in range(1, 5)}  # char-level (paper)
    # F1: collect precision/recall separately (paper's method, eval_main.ipynb cell-19)
    precisions: list[float] = []
    recalls: list[float] = []
    predictions: list[str] = []
    references: list[str] = []
    items_for_perp: list[dict] = []

    # KhanQ: evaluate on topic2/question2 columns (paper's method, eval_main.ipynb cell-7)
    topic_col = "topic2" if dataset == "khanq" else "topic"
    ref_col   = "question2" if dataset == "khanq" else "question"

    print(f"  Evaluating: {model_name} (topic={topic_col}, ref={ref_col})")

    from nltk.tokenize import word_tokenize

    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc=model_name):
        topic   = str(row[topic_col])
        context = str(row["text"])
        ref     = str(row[ref_col])

        generated = model_obj.generate_question(topic, context)
        if generated in ("[ERROR]", "[TIMEOUT]", ""):
            continue

        predictions.append(generated)
        references.append(ref)
        items_for_perp.append({"topic": topic, "text": context, "question": ref})

        b1, b2, b3, b4 = bleu_ngrams(ref, generated)
        bleu_w["bleu1"].append(b1)
        bleu_w["bleu2"].append(b2)
        bleu_w["bleu3"].append(b3)
        bleu_w["bleu4"].append(b4)

        c1, c2, c3, c4 = bleu_ngrams_char(ref, generated)
        bleu_c["bleu1_char"].append(c1)
        bleu_c["bleu2_char"].append(c2)
        bleu_c["bleu3_char"].append(c3)
        bleu_c["bleu4_char"].append(c4)

        # Collect per-sample precision/recall (not per-sample F1)
        gen_toks = set(word_tokenize(generated.lower()))
        ref_toks = set(word_tokenize(ref.lower()))
        if gen_toks and ref_toks:
            common = gen_toks & ref_toks
            precisions.append(len(common) / len(gen_toks))
            recalls.append(len(common) / len(ref_toks))

    def _mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    # Paper's corpus-level F1: harmonic mean of (mean_precision, mean_recall)
    mp = _mean(precisions)
    mr = _mean(recalls)
    f1 = 2 * mp * mr / (mp + mr) if mp + mr > 0 else 0.0

    metrics: dict = {
        "model":       model_name,
        "num_samples": len(predictions),
        # Word-level BLEU (correct)
        **{k: _mean(v) for k, v in bleu_w.items()},
        # Char-level BLEU (paper's original method)
        **{k: _mean(v) for k, v in bleu_c.items()},
        "f1":          f1,
        "meteor":      compute_meteor(predictions, references),
        "rouge_l":     compute_rouge_l(predictions, references),
        "predictions": predictions,
        "references":  references,
    }

    perp = compute_perplexity(model_obj, items_for_perp)
    if perp != float("inf"):
        metrics["perplexity"] = perp

    return metrics


# ---------------------------------------------------------------------------
# Dataset / model resolution
# ---------------------------------------------------------------------------


def _find_test_csv(config: PipelineConfig, dataset: str, tool: str) -> Optional[Path]:
    """
    Find the evaluation CSV for a dataset.

    KhanQ (``mixkhanq`` mode) saves ``data.csv`` — there is no train/val/test split.
    SQuAD modes save ``test.csv``.
    """
    if dataset == "khanq":
        # Primary: mixkhanq data.csv
        p = config.training_dir(dataset, "mixkhanq") / "data.csv"
        if p.exists():
            return p
        # Fallback: if the user ran baseline/mixsquad on khanq
        for mode in ("baseline", "mixsquad", "mixsquad2x"):
            p = config.training_dir(dataset, mode) / "test.csv"
            if p.exists():
                return p
        return None

    for mode in ("mixsquad", "baseline", "mixsquad2x"):
        p = config.training_dir(dataset, mode) / "test.csv"
        if p.exists():
            return p
    return None


def _resolve_models(config: PipelineConfig, models: str, dataset: str) -> list[str]:
    """
    Expand ``'all'`` into:
      1. Auto-discovered fine-tuned T5 checkpoints (models/{mode}/best_model/)
      2. Auto-discovered Ollama models (queries running Ollama server)
      3. Gemini / other models listed in config.evaluation.zero_shot_models

    Explicit comma-separated lists are returned as-is.
    """
    if models != "all":
        return [m.strip() for m in models.split(",")]

    keys: list[str] = []

    # 1. Auto-discover fine-tuned T5 checkpoints
    for mode in ("baseline", "topic", "topic2x"):
        model_dir = config.model_dir(mode) / "best_model"
        if model_dir.exists():
            keys.append(f"t5:{mode}")

    # 2. Auto-discover Ollama models from running server
    for model_key in _discover_ollama_models():
        if model_key not in keys:
            keys.append(model_key)

    # 3. Gemini / other non-auto-discoverable models from config
    for model_key in (config.evaluation.zero_shot_models or []):
        if model_key not in keys:
            keys.append(model_key)

    return keys or ["t5:topic"]  # fallback if nothing found


def _discover_ollama_models() -> list[str]:
    """Query the running Ollama server and return all available model keys."""
    import urllib.request
    import json as _json
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = _json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        if names:
            print(f"  [Ollama] discovered {len(names)} model(s): {', '.join(names)}")
        return [f"ollama:{n}" for n in names]
    except Exception as exc:
        print(f"  [Ollama] server not reachable ({exc}), skipping auto-discovery")
        return []


def _datasets(dataset: str) -> list[str]:
    return ["squad", "khanq"] if dataset == "all" else [dataset]


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------


def _load_paper_baselines(dataset: str) -> dict:
    """Load paper baselines from results/paper_baselines.json, filtered to the given dataset."""
    import json as _json
    path = Path(__file__).parents[3] / "results" / "paper_baselines.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = _json.load(f)
    prefix = f"{dataset}/"
    return {k: v for k, v in data.items() if k.startswith(prefix) and not k.startswith("_")}


def _print_table(results: dict, dataset: str = "") -> None:
    W = 152

    # Prepend paper baselines for the relevant dataset(s)
    datasets = set(k.split("/")[0] for k in results if "/" in k)
    paper: dict = {}
    for ds in sorted(datasets):
        paper.update(_load_paper_baselines(ds))

    def _row(key: str, m: dict, is_paper: bool = False) -> None:
        ppl_val = m.get("perplexity")
        ppl = f"{ppl_val:>8.3f}" if ppl_val is not None else f"{'N/A':>8}"
        # Paper baselines only have char-level BLEU; word-level shown as "---"
        if is_paper:
            b1  = f"{'---':>6}"
            b2  = f"{'---':>6}"
            b3  = f"{'---':>6}"
            b4  = f"{'---':>6}"
        else:
            b1  = f"{m.get('bleu1',  0):>6.3f}"
            b2  = f"{m.get('bleu2',  0):>6.3f}"
            b3  = f"{m.get('bleu3',  0):>6.3f}"
            b4  = f"{m.get('bleu4',  0):>6.3f}"
        print(
            f"{key:<40} "
            f"{b1} {b2} {b3} {b4}  "
            f"{m.get('bleu1_char', 0):>6.3f} "
            f"{m.get('bleu2_char', 0):>6.3f} "
            f"{m.get('bleu3_char', 0):>6.3f} "
            f"{m.get('bleu4_char', 0):>6.3f}  "
            f"{m.get('f1',      0):>6.3f} "
            f"{m.get('meteor',  0):>8.3f} "
            f"{m.get('rouge_l', 0):>8.3f}"
            f"{ppl}"
        )

    print("\n" + "=" * W)
    print("EVALUATION RESULTS")
    print("=" * W)
    print(
        f"{'Key':<40} "
        f"{'--- word-level BLEU ---':^27} "
        f"{'--- char-level BLEU (paper) ---':^31} "
        f"{'F1':>6} {'METEOR':>8} {'ROUGE-L':>8} {'PPL':>8}"
    )
    print(
        f"{'':40} "
        f"{'B1':>6} {'B2':>6} {'B3':>6} {'B4':>6}  "
        f"{'B1c':>6} {'B2c':>6} {'B3c':>6} {'B4c':>6}  "
        f"{'':>6} {'':>8} {'':>8} {'':>8}"
    )

    if paper:
        print("-" * W + "  [paper baselines]")
        for key, m in paper.items():
            _row(key, m, is_paper=True)

    print("-" * W + "  [this run]")
    for key, m in results.items():
        _row(key, m, is_paper=False)

    print("=" * W)
    print(
        "\nNote: word-level BLEU = correct tokenisation. "
        "char-level BLEU = paper's original method (buggy, for direct comparison with paper numbers). "
        "Paper baselines have no word-level BLEU (shown as ---)."
    )
