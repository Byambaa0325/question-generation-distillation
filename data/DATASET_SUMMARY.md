# Dataset Summary

All paths are relative to the project root. Pipeline stages: **convert → wikify → topics → dataset → train**.

---

## Raw

| Dataset | Records | Size | Location |
|---------|---------|------|----------|
| SQuAD 1.1 train | 87,599 QA pairs (18,896 paragraphs, 442 articles) | 29.6 MB | `data/raw/train-v1.1.json` |
| SQuAD 1.1 dev | 10,570 QA pairs (2,067 paragraphs, 48 articles) | 4.7 MB | `data/raw/dev-v1.1.json` |
| KhanQ | 1,034 STEM questions | 842 KB | `data/raw/KhanQ.json` |

---

## Processed (stages 1–3)

### Stage 1 — Convert

| File | Records | Location |
|------|---------|----------|
| SQuAD paragraphs | 18,896 | `data/processed/ready_squad_text.json` |
| SQuAD QA pairs | 87,599 | `data/processed/ready_squad_question.json` |
| KhanQ texts | 1,034 | `data/processed/ready_khanq_text.json` |
| KhanQ questions | 1,034 | `data/processed/ready_khanq_question.json` |

### Stage 2 — Wikify (Wikifier tool)

| File | Records | Location |
|------|---------|----------|
| SQuAD paragraphs | 18,896 | `data/processed/wikified_squad_text.json` |
| SQuAD QA pairs | 87,599 | `data/processed/wikified_squad_question.json` |
| KhanQ texts | 1,034 | `data/processed/wikified_khanq_text.json` |
| KhanQ questions | 1,034 | `data/processed/wikified_khanq_question.json` |

### Stage 3 — Topic enrichment

| Dataset | Tool | Enriched | Filtered | Filter path |
|---------|------|----------|----------|-------------|
| KhanQ | Wikifier | 1,034 | 1,034 | `data/processed/wikifier/enriched_khanq_filtered.json` |
| SQuAD | WAT | 87,599 | 37,498 | `data/processed/wat/enriched_squad_filtered.json` |
| KhanQ | WAT | 1,034 | 47 | `data/processed/wat/enriched_khanq_filtered.json` |

> WAT filtering drops entries where no topic could be matched to the question. KhanQ with WAT retains very few (47) due to domain mismatch.

---

## Training-ready (stage 4)

Columns in bold are the model inputs. Format: `<topic> {topic} <context> {text}` → `{question}`.

### SQuAD variants — JSON format (`id`, `input`, `target`, `topic`)

Derived from WAT-enriched SQuAD (37,498 total), split 70/15/15.

| Variant | Train | Val | Test | Total | Columns | Location |
|---------|-------|-----|------|-------|---------|----------|
| SQuAD Baseline | 26,248 | 5,624 | 5,626 | 37,498 | id, input, target | `data/training/squad/baseline/` |
| SQuAD Topic | 26,248 | 5,624 | 5,626 | 37,498 | id, input, **topic**, target | `data/training/squad/topic/` |

### KhanQ variants — JSON format (`id`, `input`, `target`, `topic`)

Derived from WAT-enriched KhanQ (47 total after filtering), split 70/15/15.

| Variant | Train | Val | Test | Total | Columns | Location |
|---------|-------|-----|------|-------|---------|----------|
| KhanQ Baseline | 32 | 7 | 8 | 47 | id, input, target | `data/training/khanq/baseline/` |
| KhanQ Topic | 32 | 7 | 8 | 47 | id, input, **topic**, target | `data/training/khanq/topic/` |

### KhanQ variants — CSV format (`text`, `topic`, `question`)

Derived from Wikifier-enriched KhanQ (1,034 total).

| Variant | Train | Val | Test | Total | Notes | Location |
|---------|-------|-----|------|-------|-------|----------|
| KhanQ Full | 508 | 109 | 109 | 726 | 70/15/15 split | `data/training/khanq_full/` |
| Final | 508 | 109 | 109 | 726 | Same as KhanQ Full | `data/final/` |
| Training Test | 129 | 28 | 28 | 185 | Smaller development subset | `data/training_test/` |
| MixKhanQ | — | — | — | 653 | Evaluation only, no split; columns include `context1/2`, `topic2`, `question2` | `data/training/khanq/mixkhanq/data.csv` |

### MixSQuAD variants — CSV format (`text`, `topic`, `question`)

Generated from WAT-enriched SQuAD (`data/processed/wat/enriched_squad_filtered.json`, 37,498 entries).

| Variant | Train | Val | Test | Total | Notes | Location |
|---------|-------|-----|------|-------|-------|----------|
| SQuAD Baseline | 26,171 | 5,608 | 5,609 | 37,388 | Token-length filtered, 70/15/15 split | `data/training/squad/baseline/` |
| MixSQuAD | 7,000 | 1,500 | 1,500 | 10,000 | 10k randomly mixed context pairs | `data/training/squad/mixsquad/` |
| MixSQuAD2X | 14,000 | 3,000 | 3,000 | 20,000 | MixSQuAD doubled (reversed context order) | `data/training/squad/mixsquad2x/` |

> Tool: **WAT** for SQuAD, **Wikifier** for KhanQ. `pipe.train(mode='topic', dataset='squad')` uses MixSQuAD.

---

## Format reference

**JSON training files** (`data/training/squad/*/`, `data/training/khanq/{baseline,topic}/`):
```json
{"id": "abc123", "input": "<topic> X <context> ...", "target": "question?", "topic": "X"}
```

**CSV training files** (`data/training/khanq_full/`, `data/training/squad/mixsquad*/`, etc.):
```
text,topic,question
"paragraph text","Topic Name","Generated question?"
```

**MixKhanQ evaluation CSV** (`data/training/khanq/mixkhanq/data.csv`):
```
text,topic,question,context1,context2,question2,topic2
```
Used by `pipe.evaluate(dataset='khanq')`. The `topic2`/`question2` columns are the paper's evaluation targets.
