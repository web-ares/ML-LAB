# Machine Learning Notebooks — Preprocessing, Encoding, Regression & Classification

A collection of four focused Jupyter Notebooks covering essential Machine Learning techniques — from raw data preprocessing and encoding to Linear Regression, Gradient Descent, and Logistic Regression classification. Built as part of coursework at Manav Rachna University.

---

## Table of Contents

- [Notebooks Overview](#notebooks-overview)
- [Datasets Used](#datasets-used)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Detailed Notebook Breakdown](#detailed-notebook-breakdown)
- [Results Summary](#results-summary)

---

## Notebooks Overview

| Notebook | Topic | Key Concepts |
|---|---|---|
| `pre_processing.ipynb` | Data Preprocessing & EDA | Missing values, visualizations, box plots, quartiles |
| `encoding.ipynb` | Feature Encoding | One-Hot Encoding with Pandas `get_dummies` |
| `linear_regression.ipynb` | Linear Regression | sklearn regression, Gradient Descent from scratch |
| `logistic_regression.ipynb` | Logistic Regression | Binary classification pipeline, metrics, confusion matrix |

---

## Datasets Used

| Dataset | Used In | Description |
|---|---|---|
| `titanic.csv` | `pre_processing.ipynb`, `logistic_regression.ipynb` | Passenger survival data (891 rows, 12 columns) |
| `Salary_Dataset.csv` | `encoding.ipynb` | Salary data with country and experience features |

---

## Requirements

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations and array handling |
| `pandas` | Data loading, manipulation, and encoding |
| `matplotlib` | Plotting charts and regression lines |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing pipelines, and metrics |

---

## How to Run

1. Clone or download this repository
2. Place the required datasets (`titanic.csv`, `Salary_Dataset.csv`) in the same directory as the notebooks
3. Install dependencies (see above)
4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open any notebook and run cells sequentially using **Shift + Enter**

> These notebooks were originally developed on **Google Colab**. To run on Colab, upload the CSV files using the file upload cell or mount Google Drive.

---

## Detailed Notebook Breakdown

---

### 1. `pre_processing.ipynb` — Data Preprocessing & EDA

**Dataset:** `titanic.csv`

Covers exploratory data analysis and visualization techniques on the Titanic dataset.

**Topics:**
- Loading and inspecting the dataset (`df.info()`, `df.head()`)
- Count plot of passengers by gender
- Pie chart showing percentage of passengers by sex
- Histogram of age distribution
- KDE (Kernel Density Estimate) distribution plot of age
- Box plot of age with annotated Q1, Median, and Q3 quartile values

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

### 2. `encoding.ipynb` — Feature Encoding

**Dataset:** `Salary_Dataset.csv`

Covers categorical feature encoding using One-Hot Encoding.

**Topics:**
- Loading the Salary dataset
- Applying `pd.get_dummies()` to the `country` column (One-Hot Encoding)
- Concatenating encoded dummy columns back into the main dataframe
- Reordering columns: `Australia`, `Canada`, `Dubai`, `USA`, `YearsExperience`, `Salary`, `Purchased`

**Libraries:** `pandas`

---

### 3. `linear_regression.ipynb` — Linear Regression & Gradient Descent

**Dataset:** Inline (CGPA vs Salary)

Covers two approaches to fitting a linear model — using scikit-learn and building Gradient Descent from scratch.

**Part A — sklearn Linear Regression:**
- Defining CGPA (X) and Salary in LPA (Y)
- Fitting a `LinearRegression` model
- Extracting slope, intercept, and printing the regression equation
- Predicting salary for a new CGPA value (8.5)
- Plotting actual data points vs the fitted regression line

**Part B — Gradient Descent from Scratch:**
- Manually implementing gradient descent
- Iterating over 1000 steps with learning rate `0.08`
- Computing MSE cost and updating slope (`m`) and intercept (`b`) at every step
- Printing convergence values per iteration

**Libraries:** `numpy`, `matplotlib`, `scikit-learn`

---

### 4. `logistic_regression.ipynb` — Logistic Regression (Binary Classification)

**Dataset:** `titanic.csv` (891 passengers, predicting survival)

Covers end-to-end binary classification using a full scikit-learn pipeline.

**Topics:**
- Loading and inspecting the Titanic dataset
- Dropping non-informative columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Building separate preprocessing pipelines:
  - Numerical: `SimpleImputer` with mean strategy
  - Categorical: `SimpleImputer` (most frequent) + `OneHotEncoder`
- Combining transformers with `ColumnTransformer`
- Full `Pipeline` with `LogisticRegression` (liblinear solver)
- 80/20 train-test split
- Evaluation metrics:
  - **Accuracy: 79.33%**
  - Precision, Recall, F1-score per class
  - Confusion Matrix: `[[89, 16], [21, 53]]`

**Libraries:** `pandas`, `scikit-learn`

---

## Results Summary

| Model | Dataset | Result |
|---|---|---|
| Linear Regression | CGPA vs Salary | Slope: 1.3 · Intercept: −4.8 |
| Gradient Descent | Synthetic (y = 2x + 3) | Converges after 1000 iterations |
| Logistic Regression | Titanic Survival | **Accuracy: 79.33%** |

---

## Author

**Aman Kumar Rai**
B.Tech Computer Science & Engineering (2nd Year)
Manav Rachna University
[GitHub](https://github.com/web-ares) · [LinkedIn](www.linkedin.com/in/aman-rai0709) · amanrai9660@gmail.com
