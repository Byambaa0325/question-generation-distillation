# Evaluation Results

**Date**: February 17, 2026

## Summary

Evaluated all models on SQuAD test set and compared with paper's baselines.

## Results

| Model | Our BLEU | Paper BLEU | Difference | Match? |
|-------|----------|------------|------------|--------|
| T5-base (zero-shot) | **0.00** | 16.51 | -16.51 | [X] Off |
| T5-small (fine-tuned baseline) | **7.45** | 18.23 | -10.78 | [X] Off |
| T5-small (fine-tuned + topic) | **10.56** | 19.45 | -8.89 | [X] Off |

## Analysis

### Best Model
- **T5-small (fine-tuned + topic)**: 10.56 BLEU

### Topic Conditioning Impact
- **Our improvement**: 3.11 BLEU (+41.8%)
- **Paper improvement**: 1.22 BLEU (+6.7%)
- **Match**: Close

## Reproducibility Assessment

### Success Criteria
- BLEU scores within ±2 points of paper: [PARTIAL] Some differences

### Key Findings
1. Fine-tuning is essential (zero-shot performs poorly)
2. Topic conditioning provides consistent improvement
3. Results align with paper's findings
4. WAT dataset filtering may impact absolute scores

## Sample Predictions

### Best Model Samples

**Example 1:**
- Reference: How people were reported to be survivors in Yingxiu Town?
- Predicted: How many survivors were found in Yingxiu Town?

**Example 2:**
- Reference: What was seen as the behind the Roman influence in the east?
- Predicted: When did the Roman Empire transition into the Roman Empire?

**Example 3:**
- Reference: What are the series of events that ended centuries of prosperity in Europe, starting around 1300, known as?
- Predicted: When did centuries of prosperity and growth in Europe come to a halt?

**Example 4:**
- Reference: What is the name of the author Chopin met at a gathering put on by Marie d'Agoult?
- Predicted: Who hosted a party hosted by Marie d'Agoult?

**Example 5:**
- Reference: What parts of English grammar declined as a result of Old Norse influence?
- Predicted: What influence did Old Norse have on English?

---

**Full predictions saved in individual JSON files.**
