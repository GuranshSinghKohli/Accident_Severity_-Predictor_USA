# 🚗 US Accident Severity Predictor (USA)

A comprehensive **data analysis and machine learning project** that analyzes large-scale US accident data and predicts accident severity using statistical analysis, clustering, and supervised ML models.

---

## 📌 Project Overview

This project processes and analyzes US accident records (2016–2023) to:
- Understand accident patterns across time, location, and conditions
- Identify high-risk scenarios using clustering
- Predict accident severity using machine learning

The project is designed to be **end-to-end**: data ingestion → preprocessing → analysis → modeling → visualization.

---

## 📥 Dataset Setup

### Dataset
- **US Accidents (2016–2023)** from Kaggle  
- Source: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

> ⚠️ **Note:**  
> Due to GitHub file size limits, raw CSV datasets are **not included** in this repository.

### Option 1: Manual Download
1. Download the dataset from Kaggle
2. Extract `US_Accidents_March23.csv`
3. Place it in the project root directory

### Option 2: Kaggle CLI (Recommended)
```bash
pip install kaggle
kaggle datasets download -d sobhanmoosavi/us-accidents --unzip
