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

### Install Dependencies:
To install the required dependencies, use the following command:
pip install -r requirements.txt
### Running the Code:
1. **Training the Model:**
To train the trigram model, provide a corpus file and run:
```python
trigram_model.py <corpus_file>
```
2. **Generating Sentences:**
Once the model is trained, you can generate random sentences based on the trigram model:
```python
model = TrigramModel(<corpus_file>)
print(model.generate_sentence(t=20))  # Generates a sentence with a max length of 20 words
```
3. **Evaluating Perplexity:**
To evaluate the model's perplexity on a test corpus, run:
```python
trigram_model.py <training_high> <training_low> <test_high> <test_low>
```
This will print the perplexity for both high and low models.

4. **Essay Scoring Experiment:**
Run the essay scoring experiment by providing paths to the training and test datasets:
```python
trigram_model.py <train_high> <train_low> <test_high> <test_low>
```
The script will calculate the accuracy of the essay scoring experiment.
