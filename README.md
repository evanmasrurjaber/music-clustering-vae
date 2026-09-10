# Music Clustering VAE

An unsupervised deep learning pipeline for clustering music tracks based on acoustic features and lyrical content. This project extracts Mel-frequency cepstral coefficients (MFCCs) from 30-second audio windows and projects them into a low-dimensional latent space using Variational Autoencoders (VAEs). 

The analysis bridges standard Western datasets with regional music by combining the GTZAN genre collection with a custom BanglaBeats dataset.
## [Project Paper](https://drive.google.com/file/d/1ygnVgW69BIYJYl_ilaGBRQmV3VqecKwJ/view?usp=drive_link)
---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models & Methodology](#models--methodology)
- [Getting Started](#getting-started)

---

## Overview

Traditional music clustering often struggles with cross-cultural genre definitions. This repository contains the data processing scripts, exploratory notebooks, and model implementations to cluster audio purely by its acoustic fingerprint (and complementary lyrical data). By evaluating different generative architectures, the project aims to identify the most robust feature extraction methods for complex, multi-modal music datasets.

---

## Project Structure

```text
music-clustering-vae/
├── data/
│   ├── audio/                  # Raw audio files (GTZAN & BanglaBeats)
│   ├── lyrics/                 # Scraped and compiled text lyrics
│   └── processed data/         # Extracted feature CSVs
│       ├── GTZAN_features_30_sec.csv
│       ├── banglabeats_features_30sec.csv
│       └── combined_mfcc(GTZAN+BanglaBeats)_30sec.csv
├── Figures/                    # Generated plots and latent space visualizations
├── notebooks/
│   ├── 30sec_mfcc_extraction(BanglaBeats).ipynb
│   ├── Exploratory(GTZAN+BanglaBeats).ipynb
│   └── Exploratory(audio+lyrics).ipynb
├── .gitignore
└── README.md
```

---

## Datasets

The project utilizes two primary data sources to test the models' generalization capabilities across different musical paradigms:
1. **GTZAN**: The standard benchmark dataset for machine listening research and music genre classification.
2. **BanglaBeats**: A curated dataset of regional Bengali tracks.
3. **Combined Pipeline**: Features are extracted using Librosa and merged into `combined_mfcc(GTZAN+BanglaBeats)_30sec.csv` to map a unified latent space during training.

---

## Models & Methodology

The project documents the methodology, feature extraction, and clustering performance across three distinct autoencoder variants:

| Architecture | Purpose |
|---|---|
| **Standard VAE** | The baseline variational autoencoder used for learning continuous latent representations of the MFCC features. |
| **ConvVAE** | Convolutional VAE designed to capture the spatial and temporal hierarchies present in the audio spectrograms more effectively than dense layers. |
| **Beta-VAE** | An extension of the standard VAE that learns highly disentangled representations by heavily penalizing the KL divergence, leading to cleaner cluster boundaries. |

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch
- Librosa, Pandas, Scikit-learn, Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/evanmasrurjaber/music-clustering-vae.git
cd music-clustering-vae

# Install dependencies (ensure you have your preferred environment active)
pip install -r requirements.txt
```

### Usage

1. **Feature Extraction**: Run `30sec_mfcc_extraction(BanglaBeats).ipynb` to process the raw audio files into MFCC CSVs.
2. **Exploratory Data Analysis**: Open `Exploratory(GTZAN+BanglaBeats).ipynb` to visualize the combined feature distributions.
3. **Multi-modal Analysis**: Use `Exploratory(audio+lyrics).ipynb` to see the integration of the text-based lyric data with the acoustic features.

---
