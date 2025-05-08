# Trigram Language Model

This project implements a Trigram Language Model in Python, using unsmoothed and smoothed probabilities to calculate sentence likelihoods and evaluate perplexity. It includes an essay scoring experiment that classifies essays into high and low quality based on perplexity values.

## Project Structure

```text
.
├── trigram_model.py         # Main Python file with the TrigramModel class
├── data/
│   ├── brown_train          # Training data from Brown corpus
│   ├── brown_test           # Test data from Brown corpus
│   └── ets_toefl_data/
│       ├── train_high.txt   # High-quality training essays
│       ├── train_low.txt    # Low-quality training essays
│       ├── test_high/       # Folder of high-quality test essays
│       └── test_low/        # Folder of low-quality test essays
```

## Features

- N-gram extraction for arbitrary `n`
- Raw unigram, bigram, and trigram probability calculation
- Smoothed trigram probability using linear interpolation
- Sentence-level log-probability and perplexity
- Essay scoring experiment using perplexity-based classification

## Data

The `data/` folder contains:

- `brown_train` and `brown_test` — used for training and evaluation on general language data
- `ets_toefl_data/` — used for essay scoring experiments:
  - `train_high.txt` and `train_low.txt`: Training data for two essay quality classes
  - `test_high/` and `test_low/`: Multiple essay files for evaluation

## Usage

Run the program in interactive mode:

```bash
python -i trigram_model.py data/brown_train
