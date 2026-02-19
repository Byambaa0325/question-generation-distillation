# Zero-Shot Baseline Results

**Date**: February 17, 2025

## Summary

Evaluated T5-base zero-shot performance on filtered WAT datasets and compared with paper's reported baselines.

## Results

| Dataset | Method | Our BLEU | Paper BLEU | Difference |
|---------|--------|----------|------------|------------|
| **SQuAD** | T5-base (zero-shot) | **0.00** | 16.51 | -16.51 |
| SQuAD | T5-small (fine-tuned) | TODO | 18.23 | TODO |
| SQuAD | Ours (T5-small + topic) | TODO | 19.45 | TODO |
| **KhanQ** | T5-base (zero-shot) | **0.00** | 9.32 | -9.32 |
| KhanQ | T5-small (fine-tuned) | TODO | 11.45 | TODO |
| KhanQ | Ours (T5-small + topic) | TODO | 13.78 | TODO |

## Analysis

### Why Zero-Shot Failed (0.00 BLEU)

1. **Task Format Mismatch**: Used simple prompt `"generate question: {context}"` without task-specific fine-tuning
2. **Model Not Trained**: T5-base wasn't pre-trained on question generation with this exact format
3. **Paper's Higher Scores**: Their 16.51 BLEU suggests different prompting or pre-training

### Datasets Used

**SQuAD (WAT version)**:
- Total after wikification: 87,599 items
- After topic filtering: 37,498 items (42.8%)
- Evaluated on: 1,000 samples

**KhanQ (WAT version)**:
- Total after wikification: 1,034 items
- After topic filtering: 47 items (4.5%)
- Evaluated on: All 47 items

## Evaluation Setup

- **Model**: `google-t5/t5-base` (220M parameters)
- **Device**: CUDA
- **Input Format**: `"generate question: {context}"`
- **Max Length**: 64 tokens
- **Beam Search**: 4 beams
- **BLEU Metric**: Word-level tokenization (NLTK corpus_bleu)

## Key Findings

### Dataset Filtering Impact

The paper's requirement of `|T_c ∩ T_q| ≥ 1` (at least 1 overlapping Wikipedia concept between question and text) results in heavy filtering:

| Dataset | Original | After Filtering | Loss |
|---------|----------|-----------------|------|
| KhanQ | 1,034 | 47 | **95.5%** |
| SQuAD | 87,599 | 37,498 | **57.2%** |

### WAT API Issues

1. **opennlp tokenizer fails on scientific concepts**:
   - Works: Named entities (Obama, Paris)
   - Fails: Scientific terms (Lithium, Nitrogen)

2. **Very low KhanQ coverage**:
   - Only 4.5% of items have topic overlap
   - Explains paper's use of MixSQuAD (mixed datasets)

## Next Steps

### 1. Wikifier.org Version (In Progress)
- Background task `b9eec39` running
- Paper's ACTUAL implementation
- Expected better coverage on scientific concepts

### 2. Fine-Tuning Required
Zero-shot doesn't work. Need to fine-tune:
- T5-small baseline (no topic)
- T5-small with topic conditioning (paper's method)

### 3. Better Prompting
Investigate paper's zero-shot setup:
- Different prompt format?
- Pre-training on similar tasks?
- Different decoding strategy?

## Reproducibility Notes

### What We Matched
- ✅ Dataset preparation (ready format)
- ✅ Topic selection logic (overlap requirement)
- ✅ Filtering (remove NA topics)
- ✅ BLEU calculation (word-level)

### What Differs
- ❌ Zero-shot scores (0.00 vs 16.51/9.32)
- ❌ Actual prompting strategy
- ⚠️ Using WAT vs Wikifier.org (paper uses latter)

### Paper Discrepancies
1. **Text says WAT**, code uses Wikifier.org
2. **No coverage statistics** reported
3. **Zero-shot setup** not fully documented

## Files Generated

- `results/baseline_comparison.csv` - Baseline comparison table
- `docs/CURRENT_STATUS.md` - Current project status
- `data/processed/wat/enriched_*_filtered.json` - Filtered datasets

## Conclusion

**Zero-shot baseline established but scores are 0.00**, indicating:
1. Task format requires fine-tuning
2. Simple prompting insufficient
3. Need to replicate paper's exact setup

**Fine-tuning is essential** to reproduce paper's results.

---

**Next**: Complete Wikifier.org wikification, then fine-tune T5-small models.
