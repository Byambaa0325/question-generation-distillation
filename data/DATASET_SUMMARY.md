# Dataset Download Summary

## Successfully Downloaded Datasets

### 1. SQuAD 1.1
- **Location**: `data/raw/train-v1.1.json` (29 MB) and `data/raw/dev-v1.1.json` (4.7 MB)
- **Source**: https://rajpurkar.github.io/SQuAD-explorer/
- **Description**: ~100,000 questions from 536 Wikipedia articles
- **Structure**:
  ```json
  {
    "data": [
      {
        "title": "Article_Title",
        "paragraphs": [
          {
            "context": "paragraph text...",
            "qas": [
              {
                "question": "question text",
                "id": "unique_id",
                "answers": [{"text": "answer", "answer_start": 123}]
              }
            ]
          }
        ]
      }
    ]
  }
  ```

### 2. KhanQ
- **Location**: `data/raw/KhanQ.json` (843 KB)
- **Source**: https://github.com/Huanli-Gong/KhanQ
- **Description**: 1,034 high-quality STEM questions from Khan Academy learners
- **Structure**:
  ```json
  [
    {
      "Source": "topic/source name",
      "Context": "context text...",
      "Prompt": {
        "type": "Question|Citation",
        "content": "prompt content"
      },
      "Question": "question text"
    }
  ]
  ```

## Wikifier API Configuration

- **Config file**: `config/wikifier_config.json`
- **API Key**: Saved and ready to use
- **API URL**: http://www.wikifier.org/annotate-article

## Next Steps

1. **Convert datasets to required format** for wikification:
   - Extract context and questions from SQuAD
   - Convert to JSON format with "text" field

2. **Run wikification pipeline**:
   - `wikify_text.ipynb` - Annotate contexts with Wikipedia concepts
   - `wikify_question.ipynb` - Annotate questions with Wikipedia concepts
   - `select_topic.ipynb` - Match concepts and select topics

3. **Create augmented datasets**:
   - `creat_dataset.ipynb` - Mix contexts, filter by token length, create final dataset

## Data Format Notes

- **SQuAD**: Hierarchical structure (article → paragraph → question)
- **KhanQ**: Flat structure with Source, Context, Prompt, Question
- Both need to be converted to common format for processing
