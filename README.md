# NLP Sentiment Chatbot

A natural language processing pipeline that benchmarks multiple machine learning classifiers for sentiment analysis using both TF-IDF and GloVe word embeddings, paired with a dialogue-based chatbot that performs real-time sentiment prediction and stylistic analysis.

## Overview

This project is split into two components:

**`sentiment_classifier.py`** — Trains and evaluates four ML classifiers across two feature representations (TF-IDF and GloVe), producing a full classification report with precision, recall, F1, and accuracy metrics for all 8 model-embedding combinations.

**`sentiment_chatbot.py`** — An interactive chatbot that predicts user sentiment from free-text input, performs dependency parsing via Stanford CoreNLP, and reports stylistic features including grammatical relation counts and average sentence length.

## Models

| Model | TF-IDF | GloVe |
|---|---|---|
| Naive Bayes | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ |
| Support Vector Machine | ✓ | ✓ |
| Multilayer Perceptron | ✓ | ✓ |

## Features

- **TF-IDF vectorization** using NLTK tokenization
- **300-dimensional GloVe embeddings** (pretrained on Dolma), averaged across tokens to produce document-level vectors
- **Sentiment classification** with precision, recall, F1, and accuracy evaluation
- **Dependency parsing** via Stanford CoreNLP to extract nominal subjects, direct/indirect objects, nominal modifiers, and adjectival modifiers
- **Custom stylistic feature**: average sentence length across user input
- **Dialogue management** with state-based chatbot flow and session logging

## Requirements

```
pandas
numpy
scikit-learn
nltk
```

For the chatbot's dependency parsing feature, a running [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) server is required at `http://localhost:9000`.

Install dependencies:
```bash
pip install pandas numpy scikit-learn nltk
```

## Setup

1. Download the pretrained GloVe embeddings file `glove.pkl` and place it in the project directory.
2. Place your training data (`dataset.csv`) and test data (`test.csv`) in the project directory. Each CSV should have a `review` column (string) and a `label` column (0 or 1).

## Usage

**Run the classifier and generate a classification report:**
```bash
python sentiment_classifier.py
```
Outputs `classification_report.csv` with results for all 8 model-embedding combinations.

**Run the chatbot:**
```bash
python sentiment_chatbot.py
```
Starts an interactive session. The chatbot will ask for your name, predict your sentiment, perform a stylistic analysis, and log the full conversation to a timestamped `.txt` file.

## Project Structure

```
nlp-sentiment-chatbot/
├── sentiment_classifier.py   # Model training, evaluation, classification report
├── sentiment_chatbot.py      # Interactive chatbot with sentiment + stylistic analysis
├── glove.pkl                 # Pretrained GloVe embeddings (not included, see Setup)
├── dataset.csv               # Training data (not included)
├── test.csv                  # Test data (not included)
└── README.md
```

## Results

After training, results are saved to `classification_report.csv`. Each row contains the model name, embedding type, and precision/recall/F1/accuracy scores on the held-out test set.
