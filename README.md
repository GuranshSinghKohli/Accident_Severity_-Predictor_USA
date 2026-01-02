# 🚗 US Accident Severity Predictor

## 📥 Data Setup

### Step 1: Download Dataset
The model requires the **US Accidents (2016 - 2023)** dataset from Kaggle:

1. **Download from Kaggle:**
   - Visit: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
   - Click **Download** button
   - Extract `US_Accidents_March23.csv` to project root

2. **OR use Kaggle CLI (Automated):**
   ```bash
   pip install kaggle
   kaggle datasets download -d sobhanmoosavi/us-accidents --unzip

A comprehensive machine learning system for predicting accident severity using real US accident data.

## 📊 Project Overview

This project analyzes US accident data and builds a predictive model to determine accident severity levels based on various factors including:
- Weather conditions
- Location
- Time of day
- Road type
- Visibility
- Number of vehicles involved

## 🎯 Features

- **Data Processing**: Clean and preprocess accident data
- **ML Model**: Random Forest classifier for severity prediction
- **Database Integration**: MySQL database for data storage
- **Visualizations**: Comprehensive charts and analysis
- **Clustering**: K-means clustering analysis

## 📦 Requirements

- Python 3.8+
- MySQL Server
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- mysql-connector-python

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
