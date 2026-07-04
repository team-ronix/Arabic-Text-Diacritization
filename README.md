# Arabic Text Diacritization

This project restores Arabic diacritics (tashkeel) for undiacritized Arabic text using deep learning.

The workflow is notebook-based:

1. Train a sequence model on fully diacritized text.
2. Run inference on undiacritized text.
3. Export character-level prediction files and reconstructed diacritized text.

## Project Overview

Arabic diacritization is treated as a character-level sequence labeling task.

For each character, the model predicts a diacritic class. The implementation combines:

- Character embeddings
- Word embeddings (aligned per character)
- Position-in-word embedding (inside word, end of word, or space)
- Bidirectional LSTM layers with a softmax output over diacritic classes

The notebooks also include DER (Diacritization Error Rate) evaluation utilities, including separate reporting for last vs non-last characters in words.

## Repository Structure

- `train.ipynb`: end-to-end training pipeline (preprocessing, feature extraction, model training, validation)
- `predict.ipynb`: inference pipeline and output generation
- `data/`: train/validation/test text files and sample test artifacts
- `utils/`: serialized dictionaries and mappings (`*.pickle`)
- `models/`: saved checkpoints and trained model files
- `Results/`: validation plots/images
- `predictions.csv`: full character-level predictions
- `predictions_case_ending.csv`: filtered predictions for case-ending subset

## Data and Features

Training and validation data are loaded from:

- `data/train.txt`
- `data/val.txt`

Prediction input is loaded from:

- `data/dataset_no_diacritics.txt`

Text preprocessing in the notebooks includes:

- Arabic text cleaning (keeps Arabic letters, known diacritics, and spaces)
- Sentence tokenization
- Splitting long segments by a configurable window size
- Character/diacritic extraction and aligned word-level features

## Requirements

Main Python dependencies used by the notebooks:

- tensorflow
- numpy
- pandas
- pyarabic
- scikit-learn

Install example:

```bash
pip install tensorflow numpy pandas pyarabic scikit-learn
```

## How to Run

### 1) Train

Open `train.ipynb` and run cells in order.

This notebook:

- Loads dictionaries from `utils/`
- Builds/updates vocabularies (for example `char2id.pickle`, `word2id.pickle`)
- Trains the BiLSTM-based model
- Saves checkpoints and model artifacts to `models/`

### 2) Predict

Open `predict.ipynb` and run cells in order.

This notebook:

- Loads dictionaries and model artifacts
- Processes undiacritized test input
- Generates character-level predictions
- Writes final outputs

## Output Files

After running prediction, expected outputs are:

- `predictions.csv`: all character-level predicted labels
- `predictions_case_ending.csv`: predictions filtered by `case_ending == TRUE`
- `diacritized_output.txt`: reconstructed diacritized text segments

## Notes

- Keep file paths in notebook constants consistent with your local setup.
- Ensure the trained model path in `predict.ipynb` points to an existing model under `models/`.
- The project currently uses notebooks as the main interface (no separate CLI script).
