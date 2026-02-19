# SQUAD Evaluation Results

**Date**: February 18, 2026

## Summary

Evaluated all models on SQUAD test set with proper topic conditioning.

### Key Setup
- **Zero-shot**: Uses specialized prompt + topic conditioning
- **Fine-tuned**: Uses topic<sep>context format
- **All models**: Include topic information

## Results

| Model | Our BLEU | Paper BLEU | Difference | Status |
|-------|----------|------------|------------|--------|
| T5-base (zero-shot + topic) | **0.00** | 16.51 | -16.51 | [X] Different |
| T5-small (fine-tuned baseline) | **6.95** | 18.23 | -11.28 | [X] Different |
| T5-small (fine-tuned + topic) | **10.56** | 19.45 | -8.89 | [X] Different |

## Analysis

### Best Model
- **T5-small (fine-tuned + topic)**: 10.56 BLEU

### Topic Conditioning Impact
- **Our improvement**: -6.95 BLEU (+-100.0%)
- **Paper improvement**: 1.22 BLEU (+6.7%)

## Sample Predictions

### T5-small (fine-tuned + topic)

**Example 1:**
- Reference: Would observing Daylight Saving Time have a small or large effect on how light it is during the workday in areas at high latitudes?
- Predicted: What is the DST of little use for for locations near the equator?

**Example 2:**
- Reference: The West, South Asia and Middle East think eating dogs is what?
- Predicted: What does Western, South Asian, African, and Middle Eastern cultures view dog meat as?

**Example 3:**
- Reference: What law regulated the rules for time shifts in Israel according to the Jewish calendar?
- Predicted: What year did Israel standardized daylight saving time according to the Gregorian calendar?

**Example 4:**
- Reference: Portugal modernized its public cultural facilities during what two decades?
- Predicted: How many UNESCO World Heritage Sites does Portugal have?

**Example 5:**
- Reference: What affected America's response to the situation in South Korea?
- Predicted: What was South Korea's national interest?

