# Chest X-ray Disease Classifier

A convolutional neural network trained to classify chest X-rays into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**. Built from scratch with iterative architecture refinement.

## Overview

Chest radiograph interpretation is a high-volume clinical task where automated pre-screening could meaningfully support radiologists. This project builds and iterates on CNN architectures for three-class classification, using a large public chest X-ray dataset (~16,000 training images).

## Approach

Four CNN architectures were trained and compared:

1. **Baseline CNN** — shallow encoder, small filters; established overfitting baseline
2. **CNN with more filters** — increased capacity; overfitting persisted, confirming the issue was architectural
3. **CNN with skip connections** — multi-scale feature pooling across encoder layers; validation loss stabilized
4. **Deeper CNN with skip connections** — 8-layer encoder, multi-scale features; best overall performance

Each model used instance normalization, LeakyReLU activations, data augmentation (flips, brightness, contrast), and a learning rate schedule. Image aspect ratio was passed as an auxiliary input feature.

## Results

| Model | Val Accuracy | Notes |
|-------|-------------|-------|
| Baseline CNN | ~42% | Severe overfitting |
| Increased filters | ~52% | Marginal gain |
| Skip connections | ~67% | Validation loss stabilized |
| **Deep + skip (final)** | **~76.7%** | Best overall |

**Test set performance (best model):**
- Overall accuracy: **76.72%**
- TB sensitivity: **>97%**
- Pneumonia sensitivity: **>99%**
- Normal specificity: **37%** ← identified weakness; model over-predicts disease

The high disease sensitivity is encouraging for a screening context (few missed cases), but normal specificity would need substantial improvement before clinical use.

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, pandas, scikit-learn, seaborn, matplotlib
- **Key concepts:** CNN architecture design, skip connections, instance normalization, class imbalance, confusion matrix analysis, data augmentation

## Context

Completed as part of an M.S. in Data Science. Dataset sourced from a public chest X-ray repository. The biomedical framing is a natural extension of my professional background in assay development and molecular diagnostics.
