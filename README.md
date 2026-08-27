# OncoScreen — Disease Prediction

> **Machine Learning–Based Breast Tumor Classification | CodeAlpha ML Internship — Task 4**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJavaScript-red.svg)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Medical%20ML-purple.svg)]()

## 📌 Overview

**OncoScreen** is a machine-learning disease prediction project developed as **Task 4 of the CodeAlpha Machine Learning Internship**.

The project uses the **Wisconsin Diagnostic Breast Cancer dataset** to classify breast tumors into **malignant** or **benign** patterns. It implements two complementary modeling approaches:

1. **Random Forest** using all 30 available features as a benchmark.
2. **Logistic Regression** using the top 8 features selected from the Random Forest, producing a smaller and more transparent model suitable for browser-based inference.

The accompanying browser application allows users to interactively modify the eight model inputs and immediately see the calculated classification and malignant probability.

The source documentation identifies the repository as `CodeAlpha_DiseasePrediction` and the project date as **July 2026**.

---

## 🎯 Project Objective

The primary objective is to demonstrate an end-to-end machine-learning workflow for medical classification:

**Medical Dataset → Data Loading → Train/Test Split → Random Forest Benchmark → Feature Importance → Feature Selection → Standardization → Logistic Regression → Evaluation → Browser Deployment**

The project focuses on demonstrating machine-learning methodology rather than creating a clinically deployable diagnostic system.

---

## 🧠 Problem Statement

Breast tumor classification is a binary classification problem in which the model attempts to distinguish between:

* **Benign**
* **Malignant**

The project uses numerical diagnostic measurements from the Wisconsin Diagnostic Breast Cancer dataset.

The Python implementation loads the dataset through `sklearn.datasets.load_breast_cancer`, performs a stratified 80/20 train-test split, and evaluates models using multiple classification metrics.

---

## 📊 Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer dataset** available through scikit-learn.

The deployed interface explicitly identifies the dataset as containing **569 samples**.

The original dataset contains **30 diagnostic features**. The project initially uses these features for the Random Forest benchmark and subsequently selects eight important features for the deployed Logistic Regression model.

### Dataset characteristics

* **Samples:** 569
* **Problem type:** Binary classification
* **Original feature count:** 30
* **Benchmark model:** Random Forest
* **Deployed model:** Logistic Regression
* **Train/test split:** 80% / 20%
* **Random state:** 42
* **Split strategy:** Stratified

---

## 🔬 Machine Learning Methodology

### 1. Dataset Loading

The implementation imports the Wisconsin breast cancer dataset using:

```python
from sklearn.datasets import load_breast_cancer
```

The feature matrix and target vector are extracted from the returned dataset object.

### 2. Train/Test Split

The dataset is divided into training and testing subsets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Using stratification helps preserve the class distribution between the training and testing subsets.

---

# 🌲 Random Forest Benchmark

The first model is a Random Forest classifier trained using all 30 features.

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
```

The model is trained and evaluated on the held-out test set.

### Why Random Forest?

Random Forest provides a useful benchmark because it can model nonlinear relationships and also provides feature-importance scores.

Those feature-importance values are subsequently used to identify the eight most influential features for the smaller deployed model.

---

# 🔎 Feature Selection

After training the Random Forest, the project extracts:

```python
rf.feature_importances_
```

The importance values are sorted and the eight highest-ranked features are selected.

The selected features are:

1. `worst perimeter`
2. `worst area`
3. `worst concave points`
4. `mean concave points`
5. `worst radius`
6. `mean radius`
7. `mean perimeter`
8. `mean area`

These eight features are used by the deployed Logistic Regression model.

---

# 📐 Feature Standardization

Because Logistic Regression is sensitive to feature scale, the selected features are standardized using:

```python
StandardScaler()
```

The scaler is fitted on the training data and then applied to both training and test data:

```python
Xtr2_s = scaler.fit_transform(Xtr2)
Xte2_s = scaler.transform(Xte2)
```

This prevents information from the test set from being used during scaler fitting.

---

# 📈 Logistic Regression

The deployed model is:

```python
LogisticRegression(max_iter=2000)
```

It is trained using the eight selected features after standardization.

The reason for deploying this smaller model instead of the full Random Forest is practical: the project aims for a **transparent, browser-sized inference implementation**.

---

# 📊 Model Evaluation

The project evaluates models using five metrics:

* **ROC-AUC**
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

These metrics are calculated through scikit-learn's classification metrics.

## Reported Results

| Model                        | Features | ROC-AUC | Accuracy | Precision | Recall |     F1 |
| ---------------------------- | -------: | ------: | -------: | --------: | -----: | -----: |
| Random Forest                |       30 |  0.9940 |   0.9474 |    0.9583 | 0.9583 | 0.9583 |
| Deployed Logistic Regression |        8 |  0.9927 |   0.9474 |    0.9714 | 0.9444 | 0.9577 |

These values are embedded in the project's deployed model configuration.

### Interpretation

The results show that the reduced eight-feature Logistic Regression model achieves performance close to the full 30-feature Random Forest benchmark on the reported test split.

This makes the smaller model attractive for demonstrating lightweight, transparent browser inference while retaining strong benchmark performance.

**Important:** these metrics represent this particular dataset and train/test split. They should not be interpreted as clinical performance or real-world diagnostic accuracy.

---

# 🌐 Interactive Web Application

The project includes:

```text
CodeAlpha_DiseasePrediction.html
```

The interface is called **OncoScreen** and provides a compact browser-based prediction interface.

The application includes:

* Eight interactive sliders
* Live model inference
* Malignant probability
* Benign/malignant visual state
* Probability gauge
* Precision display
* Recall display
* F1-score display
* ROC-AUC display
* Benign-like sample preset
* Malignant-like sample preset

The two preset controls are explicitly implemented in the HTML application.

---

# ⚙️ Browser Inference

One of the notable aspects of this project is that the browser application contains the trained Logistic Regression parameters directly.

The JavaScript performs the same basic mathematical transformation used during model inference:

1. Read the selected feature value.
2. Standardize it using the stored mean and scale.
3. Multiply by the corresponding Logistic Regression coefficient.
4. Add the intercept.
5. Apply the sigmoid function.
6. Convert the result into a probability.
7. Display the classification.

The implementation calculates the malignant probability from the model's stored coefficients and standardized inputs.

This means the demonstration does not require a Python server just to perform the browser prediction.

---

# 🖥️ Prediction Output

The interface displays one of two primary outcomes:

### Benign Pattern

The application displays:

```text
Benign pattern
```

when the calculated malignant probability is at or below the model's classification threshold.

### Malignant Risk

The application displays:

```text
Malignant risk
```

when the malignant probability exceeds the threshold used by the browser implementation.

The application also displays the calculated malignant probability as a percentage.

---

# 🧪 Example Presets

For demonstration purposes, the application generates two preset input configurations:

* **Benign-like sample**
* **Malignant-like sample**

These are generated from the defined feature ranges rather than representing actual individual patient records.

This distinction is important: the presets are demonstration inputs and should not be interpreted as clinical patient examples.

---

# 📁 Project Structure

A recommended repository structure is:

```text
CodeAlpha_DiseasePrediction/
│
├── disease_prediction.py
├── CodeAlpha_DiseasePrediction.html
├── CodeAlpha_DiseasePrediction_ResearchPaper.pdf
├── README.md
└── LICENSE
```

The source documentation identifies both the Python training/evaluation source and the deployed HTML tool as part of the project.

---

# 🛠️ Technologies Used

### Programming

* Python
* JavaScript
* HTML
* CSS

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* StandardScaler

### Numerical Computing

* NumPy

### Evaluation

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-Score

The Python source explicitly imports NumPy and the relevant scikit-learn components for model training, preprocessing, and evaluation.

---

# 🚀 Running the Python Model

Install the required Python packages:

```bash
pip install numpy scikit-learn
```

Then run:

```bash
python disease_prediction.py
```

The script loads the dataset, trains the Random Forest benchmark, extracts the top eight features, trains the Logistic Regression model, and prints the evaluation metrics.

---

# 🌐 Running the Web Application

Because the deployed interface is an HTML document containing its model parameters and JavaScript inference logic, it can be opened directly in a modern web browser:

```text
CodeAlpha_DiseasePrediction.html
```

For GitHub Pages deployment, place the HTML file in the repository and configure GitHub Pages for the repository.

---

# 🔄 End-to-End Pipeline

```text
Wisconsin Diagnostic Breast Cancer Dataset
                │
                ▼
        Load 30 Features
                │
                ▼
       Stratified Train/Test Split
                │
                ▼
      Random Forest Benchmark
                │
                ▼
       Feature Importance
                │
                ▼
       Select Top 8 Features
                │
                ▼
        StandardScaler
                │
                ▼
      Logistic Regression
                │
                ▼
       Model Evaluation
                │
                ▼
      Browser-Sized Model
                │
                ▼
      Interactive HTML Tool
                │
                ▼
    Malignant Probability Output
```

---

# 🔐 Reproducibility

The project uses `random_state=42` for the dataset split and Random Forest configuration. This provides deterministic behavior for the demonstrated experiment when the same software environment and dataset are used.

The model's scaler means, scaler scales, coefficients, intercept, feature ranges, and evaluation metrics are embedded in the browser implementation.

---

# ⚠️ Medical & Research Disclaimer

**OncoScreen is an educational machine-learning internship project and is NOT a medical diagnostic device.**

The application should not be used to:

* Diagnose cancer
* Replace a physician
* Make treatment decisions
* Determine whether a person has cancer
* Provide medical advice
* Replace laboratory testing, imaging, pathology, or clinical assessment

A machine-learning prediction from this project should therefore be understood strictly as a **demonstration of binary classification on a public dataset**, not as a clinical diagnosis.

The deployed application itself identifies the work as an educational internship deliverable and explicitly states that it is not a diagnostic device.

---

# 🔬 Research Value

This project demonstrates several important machine-learning concepts in a single workflow:

* Binary classification
* Medical-data modeling
* Benchmark model development
* Feature importance
* Feature selection
* Dimensionality reduction
* Feature standardization
* Logistic Regression
* Random Forest
* Probability estimation
* Model evaluation
* Browser-based inference
* Model transparency
* Lightweight deployment

A particularly useful design decision is the separation between the **full-feature benchmark** and the **reduced deployed model**. The Random Forest is used to identify important variables, while Logistic Regression provides a compact model whose parameters can be directly implemented in JavaScript.

---

# 📚 Project Deliverables

The project includes:

### `disease_prediction.py`

Responsible for:

* Dataset loading
* Train/test splitting
* Random Forest training
* Feature-importance analysis
* Top-eight feature selection
* Standardization
* Logistic Regression training
* Model evaluation

### `CodeAlpha_DiseasePrediction.html`

Responsible for:

* Interactive user interface
* Feature sliders
* Browser-side model inference
* Probability calculation
* Result visualization
* Demonstration presets
* Model-performance display

### `CodeAlpha_DiseasePrediction_ResearchPaper.pdf`

Companion research documentation referenced by the project source listing.

---

# 👨‍💻 Author

**Mayank Swaraj (Mayur Krishna)**

Machine Learning Internship Project
**CodeAlpha — Task 4**

Repository:

```text
CodeAlpha_DiseasePrediction
```

Project Date:

```text
July 2026
```

The source documentation attributes the project to Mayank Swaraj (Mayur Krishna).

---

# ⭐ Key Takeaway

**OncoScreen demonstrates how a relatively complex machine-learning benchmark can be distilled into a smaller, interpretable model and deployed directly in a browser.**

Rather than stopping at model training, the project connects **dataset → modeling → feature selection → evaluation → deployment**, creating a complete educational machine-learning workflow.

---

## 📜 License

Add an appropriate open-source license to the repository if you intend to distribute the source code publicly.

---

## ⚠️ Final Disclaimer

**For educational and research demonstration only. Not for clinical use, diagnosis, screening, treatment, or medical decision-making.**
