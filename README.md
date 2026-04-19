# Machine Learning Lab Notebook

A hands-on collection of Machine Learning concepts and implementations covering NumPy fundamentals, data preprocessing, exploratory data analysis, and core ML algorithms — built as part of coursework at Manav Rachna University.

---

## Table of Contents

- [Overview](#overview)
- [Topics Covered](#topics-covered)
- [Datasets Used](#datasets-used)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Notebook Structure](#notebook-structure)

---

## Overview

This Jupyter Notebook (`ML.ipynb`) walks through foundational Machine Learning topics from scratch. Each section focuses on a specific concept, with working code, real datasets, and visualizations to reinforce understanding.

---

## Topics Covered

### 1. NumPy Array Operations
- Creating and modifying integer arrays
- Array statistics: size, dtype, mean (median), min, max, sum, product
- Statistical measures: variance and covariance

### 2. Data Preprocessing with Pandas
- Loading a CSV dataset with custom headers (Automobile dataset)
- Handling missing values (`?` → `NaN`)
- Replacing NaN with column mean
- Dropping rows with missing target values
- Checking and auditing missing data across all columns

### 3. Exploratory Data Analysis (EDA) — Titanic Dataset
- Survival count plots
- Gender distribution bar charts
- Survival percentage pie chart
- Age distribution histogram
- Age distribution with KDE (Kernel Density Estimate)
- Box plots: Age vs Survival, Fare vs Passenger Class

### 4. Linear Regression
- Predicting salary from CGPA using `sklearn.linear_model.LinearRegression`
- Fitting the model, extracting slope and intercept
- Predicting for new input values
- Scatter plot visualization

### 5. K-Nearest Neighbours (KNN) Classification
- Classifying student pass/fail based on hours studied and hours slept
- Training a KNN classifier (`n_neighbors=3`)
- Predicting for a new data point
- Visualizing decision boundaries using `matplotlib` and `ListedColormap`

### 6. Decision Tree (ID3 / Entropy)
- Manual implementation of entropy and information gain calculation
- Finding the best splitting feature using information gain
- Training a `DecisionTreeClassifier` with entropy criterion on the Tennis dataset
- Visualizing the full decision tree with `plot_tree`
- Label encoding all categorical features
- Making predictions on new weather condition inputs

---

## Datasets Used

| Dataset | Description | Source |
|---|---|---|
| `car_dataset.data` | Automobile data with 26 features including price, engine specs, and fuel type | UCI ML Repository |
| `titanic.csv` | Passenger survival data from the Titanic | Kaggle / public domain |
| `Tennis.csv` | Weather conditions and play tennis decision (categorical) | Classic ML teaching dataset |

---

## Requirements

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `numpy` | Array operations and statistics |
| `pandas` | Data loading and preprocessing |
| `matplotlib` | Plotting and visualizations |
| `seaborn` | Statistical data visualizations |
| `scikit-learn` | ML models (Linear Regression, KNN, Decision Tree) |

---

## How to Run

1. Clone or download this repository
2. Place the required datasets (`car_dataset.data`, `titanic.csv`, `Tennis.csv`) in the same directory as the notebook
3. Install dependencies (see above)
4. Launch Jupyter Notebook:

```bash
jupyter notebook ML.ipynb
```

5. Run cells sequentially using **Shift + Enter**

---

## Notebook Structure

```
ML.ipynb
│
├── Section 1 — NumPy Basics
│   └── Array creation, modification, statistics
│
├── Section 2 — Data Preprocessing (Automobile Dataset)
│   └── Missing value handling, imputation, auditing
│
├── Section 3 — EDA (Titanic Dataset)
│   └── Count plots, pie charts, histograms, box plots
│
├── Section 4 — Linear Regression
│   └── CGPA → Salary prediction
│
├── Section 5 — KNN Classification
│   └── Student pass/fail prediction + decision boundary visualization
│
└── Section 6 — Decision Tree (Tennis Dataset)
    ├── Manual entropy & information gain calculation
    ├── sklearn DecisionTreeClassifier
    └── Prediction on new input
```

---

## Author

**Aman Kumar Rai**  
B.Tech Computer Science & Engineering (2nd Year)  
Manav Rachna University  
[GitHub](https://github.com/web-ares) · [LinkedIn](https://www.linkedin.com/in/aman-rai-75968b229) · amanrai9660@gmail.com
