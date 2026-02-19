# Google Colab Training Guide

## Quick Start

### 1. Upload Notebook to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `T5_Question_Generation_Training.ipynb`
4. Enable GPU: **Runtime → Change runtime type → GPU** (select T4, V100, or A100)

### 2. Prepare Your Data

You need these files from your local training data:

**For Baseline Model:**
```
data/training/squad/baseline/squad_train.json
data/training/squad/baseline/squad_val.json
data/training/squad/baseline/squad_test.json
```

**For Topic-Conditioned Model:**
```
data/training/squad/topic/squad_train.json
data/training/squad/topic/squad_val.json
data/training/squad/topic/squad_test.json
```

### 3. Run the Notebook

Simply run all cells in order:
1. ✅ Setup environment (check GPU)
2. ✅ Mount Google Drive
3. ✅ Upload training data
4. ✅ Train model
5. ✅ Evaluate on test set
6. ✅ Download results

Everything is saved to Google Drive automatically!

## Expected Training Times

### Free Colab (T4 GPU ~16GB)
- **Batch size**: 16
- **Time per epoch**: ~20-25 minutes
- **Total (3 epochs)**: ~60-75 minutes
- **Cost**: FREE

### Colab Pro (V100 GPU ~16GB)
- **Batch size**: 32
- **Time per epoch**: ~10-15 minutes
- **Total (3 epochs)**: ~30-45 minutes
- **Cost**: $10/month

### Colab Pro+ (A100 GPU ~40GB)
- **Batch size**: 64
- **Time per epoch**: ~5-8 minutes
- **Total (3 epochs)**: ~15-25 minutes
- **Cost**: $50/month

## Configuration Options

Edit the `CONFIG` cell to customize:

```python
CONFIG = {
    'model_name': 't5-small',        # or 't5-base' for better quality
    'batch_size': 16,                # Increase to 32/64 if you have V100/A100
    'epochs': 3,                     # Paper uses 3-5 epochs
    'learning_rate': 3e-4,           # Standard T5 learning rate
    'max_input_length': 512,         # Max context tokens
    'max_target_length': 64,         # Max question tokens
}
```

### Optimizations for Better GPUs

If you have **V100 or A100**:
```python
CONFIG = {
    'batch_size': 32,  # or 64 for A100
    'epochs': 5,       # More epochs for better convergence
}
```

## Training Both Models

### Train Baseline First
1. Upload baseline data (context only)
2. Run all cells
3. Note the BLEU score

### Then Train Topic-Conditioned
1. Upload topic data (topic`<sep>`context format)
2. Update config:
   ```python
   CONFIG = {
       'train_file': f'{PROJECT_DIR}/data/squad_train_topic.json',
       'val_file': f'{PROJECT_DIR}/data/squad_val_topic.json',
       'output_dir': f'{PROJECT_DIR}/models/t5small_topic',
   }
   ```
3. Run cells 5-7 again (skip setup cells)

## Files Generated in Google Drive

After training, you'll find:

```
MyDrive/T5_Question_Generation/
├── models/
│   └── t5small_trained/
│       ├── best_model/              ← Use this for inference
│       ├── final_model/
│       ├── checkpoint-epoch-1/
│       ├── checkpoint-epoch-2/
│       ├── checkpoint-epoch-3/
│       └── training_history.png     ← Loss curves
├── results/
│   ├── evaluation_results.json      ← BLEU score & metrics
│   └── predictions.json             ← All test predictions
└── data/
    ├── squad_train.json
    ├── squad_val.json
    └── squad_test.json
```

## Expected Results

### Paper's Baselines (SQuAD)
| Method | BLEU Score |
|--------|------------|
| T5-base (zero-shot) | 16.51 |
| T5-small (baseline) | **18.23** |
| T5-small (+ topic) | **19.45** |

### Your Results Should Be:
- **Baseline**: 17-19 BLEU (close to 18.23)
- **Topic-conditioned**: 18-20 BLEU (close to 19.45)

If your scores are within ±1-2 BLEU, you've successfully reproduced the paper! 🎉

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
CONFIG['batch_size'] = 8  # or even 4
```

### Training Too Slow
- Upgrade to Colab Pro for V100 GPU
- Or use smaller sample for testing:
  ```python
  # In "Generate questions" cell, add:
  test_data = test_data[:1000]  # Use 1000 samples instead of all
  ```

### Can't Upload Large Files
Files over 100MB may fail. Instead:
1. Upload to Google Drive directly
2. Modify the upload cell to read from Drive:
   ```python
   # Skip upload cell, data already in Drive
   CONFIG['train_file'] = '/content/drive/MyDrive/your_data/squad_train.json'
   ```

## Using the Trained Model

After training, download the best model and use locally:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
model = T5ForConditionalGeneration.from_pretrained('path/to/best_model')
tokenizer = T5Tokenizer.from_pretrained('path/to/best_model')

# Generate question
context = "Albert Einstein was a German-born theoretical physicist."
inputs = tokenizer(context, return_tensors='pt')
outputs = model.generate(inputs.input_ids, max_length=64, num_beams=4)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated: {question}")
# Expected: "Who was Albert Einstein?"
```

## Comparison: Local vs Colab

| Aspect | Local Training | Google Colab |
|--------|---------------|--------------|
| **Cost** | GPU required | Free (T4) or $10-50/month |
| **Setup** | Manual | Automated in notebook |
| **Speed** | Depends on GPU | T4: ~1 hour, A100: ~15 min |
| **Data** | Already local | Need to upload |
| **Storage** | Local disk | Google Drive (15GB free) |
| **Best for** | Multiple experiments | One-off training |

## Tips

1. **Save checkpoints**: The notebook saves after each epoch, so you won't lose progress if disconnected
2. **Monitor training**: Watch the loss decrease - should go from ~4 to ~2
3. **Use best model**: Always use `best_model/` (lowest validation loss), not `final_model/`
4. **Compare results**: Run both baseline and topic models to see the impact of conditioning

## Next Steps

After training both models:
1. Compare BLEU scores with paper
2. Analyze sample predictions
3. Test on KhanQ dataset (cross-domain evaluation)
4. Write reproducibility report

## Questions?

Check these resources:
- Paper: "Topic-Controllable Question Generation"
- Hugging Face T5 docs: https://huggingface.co/docs/transformers/model_doc/t5
- Project README: `../docs/README.md`

Happy training! 🚀
