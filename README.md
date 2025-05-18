# Trigram Language Model

This project implements a Trigram Language Model in Python, using unsmoothed and smoothed probabilities to calculate sentence likelihoods and evaluate perplexity. It includes an essay scoring experiment that classifies essays into high and low quality based on perplexity values.

## Project Structure

```text
.
├── trigram_model.py         # Main Python file with the TrigramModel class and experiment logic
├── data/
|  ├── train_high.txt   # High-quality training essays
|  ├── train_low.txt    # Low-quality training essays
|  ├── test_high/       # Folder of high-quality test essays
|  └── test_low/        # Folder of low-quality test essays
├── requirements.txt  # Dependency list (currently empty — only standard libraries used)
└── .gitignore  # Files and folders to exclude from Git version control
```

## Features

- Trigram probability estimation with linear interpolation smoothing
- Log-probability and perplexity calculation for each sentence
- Essay scoring based on average perplexity across models
- Simple toy dataset to demonstrate the concept


## Usage

### Run Essay Scoring Experiment:
From the project root, run:
```python
python trigram_model.py data/train_high.txt data/train_low.txt data/test_high data/test_low
```
This will:
- Train a trigram model on each training set
- Compute perplexity for each test essay under both models
- Predict class based on which model assigns lower perplexity
- Print overall classification accuracy
  
### Requirements:
- Python 3.x
- No external libraries required (uses only standard Python modules)

### Notes:
- This is a toy example for demonstration and learning purposes.
- Accuracy may vary due to small and noisy sample data.
- For real-world usage, better preprocessing, larger datasets, and evaluation metrics are recommended.
