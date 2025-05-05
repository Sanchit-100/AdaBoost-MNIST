# AdaBoost from Scratch on MNIST (Digits 0 and 1)

This repository contains a NumPy-based implementation of the AdaBoost algorithm for binary classification using decision stumps as weak learners. The algorithm is evaluated on a reduced MNIST dataset containing only digits `0` and `1`.

---

## 🔍 Overview

This project demonstrates:

- Implementation of AdaBoost with decision stumps (depth-1 decision trees)
- Weighted 0-1 loss and adaptive sample weighting
- Boosted prediction rule using a weighted sum of classifiers
- Evaluation on PCA-reduced MNIST dataset (2-class: 0 vs 1)
- Accuracy and error visualization across boosting rounds

---

## 🧠 Model Features

- **Base learner:** Custom decision stump with uniform threshold sampling (3 thresholds per dimension)
- **Boosting strategy:** AdaBoost (Freund & Schapire, 1997)
- **Loss metric:** Weighted 0-1 loss for misclassification
- **Final hypothesis:** Sign of weighted sum of stump predictions
- **Boosting rounds:** Up to 200 iterations
- **Prediction:** Combined score mapped from {-1, +1} to {0, 1}

---

## 🧪 Dataset

- **Source:** [MNIST](https://www.openml.org/d/554) via `sklearn.datasets.fetch_openml`
- **Classes used:** `0` and `1` only
- **Training data:** 1000 samples per class (total = 2000)
- **Test data:** Remaining samples from both classes
- **Dimensionality reduction:** PCA to 5 principal components

---

## 📊 Training Details

- Training and testing errors are tracked over 100–200 boosting rounds.
- Decision stumps evaluate 3 thresholds per PCA dimension, spaced uniformly between min and max.

---

## 📈 Results

- **Final Test Accuracy:** **99.30 \%**


## 📌 Notes

- Decision stump logic evaluates each feature independently.
- Stumps split using thresholds uniformly spaced between feature min and max values.
- Label encoding: `{0,1}` is internally mapped to `{-1,+1}` for AdaBoost math.

---

## 📋 Usage

```bash
pip install scikit-learn matplotlib
python adaboost_mnist.py
