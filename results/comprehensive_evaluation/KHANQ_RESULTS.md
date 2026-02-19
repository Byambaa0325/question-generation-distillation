# KHANQ Evaluation Results

**Date**: February 18, 2026

## Summary

Evaluated all models on KHANQ test set with proper topic conditioning.

### Key Setup
- **Zero-shot**: Uses specialized prompt + topic conditioning
- **Fine-tuned**: Uses topic<sep>context format
- **All models**: Include topic information

## Results

| Model | Our BLEU | Paper BLEU | Difference | Status |
|-------|----------|------------|------------|--------|
| T5-base (zero-shot + topic) | **0.00** | 9.32 | -9.32 | [X] Different |
| T5-small (fine-tuned baseline) | **1.59** | 11.45 | -9.86 | [X] Different |
| T5-small (fine-tuned + topic) | **2.23** | 13.78 | -11.55 | [X] Different |

## Analysis

### Best Model
- **T5-small (fine-tuned + topic)**: 2.23 BLEU

### Topic Conditioning Impact
- **Our improvement**: -1.59 BLEU (+-100.0%)
- **Paper improvement**: 2.33 BLEU (+20.3%)

## Sample Predictions

### T5-small (fine-tuned + topic)

**Example 1:**
- Reference: What is that extra H+ doing there on the head of water?
- Predicted: What does H+ come from?

**Example 2:**
- Reference: Why would an electron fall back to any energy level other than the n=1, since there are no other electrons stopping it from falling there?
- Predicted: What does an electron in the n=1 state have?

**Example 3:**
- Reference: How magnetic force when perpendicular to the velocity gives rise to circular motion?
- Predicted: What does Uniform Circular Motion mean?

**Example 4:**
- Reference: How many chromosomes would a person have?
- Predicted: How many sets of chromosomes do individuals possess?

**Example 5:**
- Reference: How an excess in prolactin can effect men's fertility?
- Predicted: What hormone makes up the amount of FSH and LH made by the pituitary?

