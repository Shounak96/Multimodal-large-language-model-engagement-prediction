# Multimodal Engagement Prediction (PixelRec)

A research-oriented machine learning project that predicts **social media engagement and content popularity** using **multimodal models** combining visual and textual signals. The project evaluates and compares **deep multimodal regression**, **feature-based classification**, and **end-to-end vision–language models** on the PixelRec dataset.

---

## Project Overview

This project investigates how **images + text** can be jointly modeled to predict social media engagement (likes, comments, shares, views, favorites). Three complementary modeling paradigms are implemented and evaluated:

1. **Multimodal Regression** using SigLIP + DeBERTa
2. **Feature-Based Classification** using CLIP embeddings + CatBoost
3. **End-to-End Classification** using ViT + BERT

The goal is to understand the strengths and limitations of each approach for predicting engagement and content popularity.

---

## Features Implemented

- Multimodal data preprocessing and normalization
- Engagement score construction and binning
- End-to-end multimodal classification (ViT + BERT)
- Feature-based classification with CatBoost
- Multimodal regression with cross-modal attention
- Comparative evaluation using accuracy, F1-score, and R²
- Experiment tracking and result logging
- Research-style report and presentation

---

## Models & Techniques Used

### Multimodal Regression
- SigLIP (vision encoder)
- DeBERTa (text encoder)
- Cross-modal attention + gated fusion
- Hybrid loss (MSE + Huber)
- R²-based model selection

### Feature-Based Classification
- CLIP image & text embeddings
- Engineered engagement and metadata features
- CatBoost classifier (GPU-accelerated)

### End-to-End Classification
- Vision Transformer (ViT)
- BERT (text encoder)
- Multimodal fusion strategies
- Five-class ordinal engagement prediction

---

## Technologies Used

- **Python**
- **PyTorch**
- **HuggingFace Transformers**
- **CatBoost**
- **Jupyter Notebooks**
- **NumPy / Pandas**
- **Scikit-learn**
- **Matplotlib**
- **PixelRec Dataset**

---

## Project Structure

├── CatBoost/                 # Feature-based classification experiments
├── Regression/               # SigLIP + DeBERTa regression models
├── ViT+BERT/                 # End-to-end multimodal classification
├── data_binner.ipynb         # Engagement score binning
├── data_prep_initial.ipynb  # Dataset preprocessing
├── fineCLIP.py               # CLIP-based feature extraction
├── siglipdeberta.out         # Training logs / outputs
├── Project_Report.pdf        # Final research report
├── Project_Presentation.pdf # Slides
└── README.md
