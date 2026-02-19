# KhanQ Evaluation Results - Concise Prompt

**Date**: February 18, 2026

## Evaluation Setup

**Zero-shot prompt:** Concise format (8-12 words max)
- Emphasizes brevity and directness
- Specific question patterns provided
- Multiple examples given

**Fine-tuned models:** topic<sep>context format

## Results

| Model | Our BLEU | Paper BLEU | Difference | Status |
|-------|----------|------------|------------|--------|
| T5-base (zero-shot + concise prompt) | **0.00** | 9.32 | -9.32 | [X] Different |
| T5-small (fine-tuned baseline) | **1.59** | 11.45 | -9.86 | [X] Different |
| T5-small (fine-tuned + topic) | **2.23** | 13.78 | -11.55 | [X] Different |

## Analysis

### Best Model
- **T5-small (fine-tuned + topic)**: 2.23 BLEU

### Topic Conditioning Impact
- **Our improvement**: 0.64 BLEU (+40.4%)
- **Paper improvement**: 2.33 BLEU (+20.3%)

### Zero-Shot Performance
- **Our score**: 0.00 BLEU
- **Paper score**: 9.32 BLEU
- **Difference**: -9.32

## Sample Predictions

### T5-base (zero-shot + concise prompt)

**Example 1:**
- Reference: How many chromosomes would a person have?
- Predicted: True

**Example 2:**
- Reference: Why do lone pairs of electrons form hybrid orbitals?
- Predicted: True

**Example 3:**
- Reference: Is the oxygen and the hydrogen just linear?
- Predicted: True

**Example 4:**
- Reference: Is there any specific reason mole was defined using the number of elemental entities of carbon-12?
- Predicted: True

**Example 5:**
- Reference: Does the additional time possible allow aging and deterioration of the protein to be processes?
- Predicted: True

### T5-small (fine-tuned baseline)

**Example 1:**
- Reference: Would I be correct to describe geraniol as 2 isoprene units and an alcohol?
- Predicted: What is the skeleton of geraniol?

**Example 2:**
- Reference: Is a non spontaneous reaction always a reverse reaction?
- Predicted: What is the sign of the gibbs?

**Example 3:**
- Reference: Doesn't that violate Markovnikov's rule that the more substituted C will bond with the more substituted reactant?
- Predicted: What is Markovnikov's rule?

**Example 4:**
- Reference: How is a ligament tear different from a sprain and why does it require surgery?
- Predicted: What is the result of capillary dilation?

**Example 5:**
- Reference: How is it that these 3 parts (electron, proton, and neutron) can combine into different things just based on how many of these 'parts' there are?
- Predicted: What is a better analogy than mixing paint?

### T5-small (fine-tuned + topic)

**Example 1:**
- Reference: How  between stable and unstable isotop differs in radiation?
- Predicted: What kind of radiation does Stable isotopes have?

**Example 2:**
- Reference: Does binary fission occur in the same way?
- Predicted: How many types of binary fission are there in bacteria?

**Example 3:**
- Reference: Will it work until all reactants are turned into products?
- Predicted: What does a catalyst increase in the rate of reaction?

**Example 4:**
- Reference: Is it in the form of particles or waves?
- Predicted: What does Thomas Young's double slit experiment mean?

**Example 5:**
- Reference: If something is reactive, does that automatically make it acidic?
- Predicted: What does the atom take on the property of being acidic?

