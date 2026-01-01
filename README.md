# Real-Time Fraud Detection using AI and SMOTE

Copyright (c) 2026 Shrikara Kaudambady. All rights reserved.

## 1. Introduction

Real-time fraud detection is a critical challenge for financial institutions. The goal is to identify and block fraudulent transactions as they happen, without inconveniencing legitimate customers. This is a classic "needle in a haystack" problem because fraudulent transactions are extremely rare compared to the vast number of legitimate ones.

This project provides a Jupyter Notebook that implements a complete machine learning solution for this problem. It demonstrates how to train a high-performance classification model on a heavily imbalanced dataset and evaluates it using appropriate metrics that reflect real-world performance.

## 2. The Solution Explained: Classification with Imbalance Handling

A naive model trained on an imbalanced dataset will achieve high accuracy simply by always predicting "not fraud," making it useless. Our solution addresses this head-on with a specialized technique and a robust evaluation framework.

### 2.1 The Challenge: Severe Class Imbalance

The public credit card fraud dataset used in this notebook has over 280,000 transactions, of which only ~0.17% are fraudulent. This imbalance means:
*   Standard **accuracy** is a misleading metric. A model that is 99.8% accurate might be catching zero fraudulent transactions.
*   The model will be heavily biased towards the majority class (legitimate transactions) and will likely fail to learn the patterns of the minority class (fraud).

### 2.2 The Technique: SMOTE (Synthetic Minority Over-sampling Technique)

To solve the imbalance problem, we use a powerful over-sampling technique called SMOTE, provided by the `imbalanced-learn` library.

**How SMOTE Works:** Instead of simply duplicating the rare fraud examples, SMOTE intelligently generates new, **synthetic** fraud examples. It does this by:
1.  Finding a fraudulent transaction in the dataset.
2.  Identifying its nearest neighbors (other similar fraudulent transactions).
3.  Creating a new, synthetic data point somewhere along the line connecting the transaction and its neighbors.

This process populates the "feature space" with more diverse, plausible examples of fraud, giving the model a much richer and more balanced dataset to learn from. **Crucially, SMOTE is applied only to the training data**, ensuring that the test data remains a true representation of the imbalanced real world.

### 2.3 The Model and Evaluation

*   **Model:** We use a **LightGBM (LGBMClassifier)**, a high-performance gradient boosting framework known for its speed and accuracy, which is essential for a "real-time" use case.
*   **Evaluation Metrics:** We discard accuracy and instead focus on metrics that are critical for fraud detection:
    *   **Precision:** What percentage of transactions we flagged as fraud were actually fraudulent? (Minimizes false positives, i.e., blocking legitimate customers).
    *   **Recall:** What percentage of all actual fraudulent transactions did we successfully catch? (Maximizes fraud detection).
    *   **Area Under the Precision-Recall Curve (AUPRC):** The gold-standard metric for imbalanced classification. It provides a single score that summarizes the trade-off between precision and recall across all possible thresholds. A higher AUPRC means a better model.

## 3. How to Use the Notebook

### 3.1. Prerequisites

This project requires several data science libraries, including `lightgbm` for the model and `imbalanced-learn` for SMOTE.

```bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn matplotlib seaborn
```

### 3.2. Running the Notebook

1.  Clone this repository:
    ```bash
    git clone https://github.com/shrikarak/real-time-fraud-detection-ai.git
    cd real-time-fraud-detection-ai
    ```
2.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
3.  Open `fraud_detection_model.ipynb` and run the cells sequentially. The notebook will load the data, apply SMOTE, train the model, and print a detailed evaluation report.

## 4. Deployment for "Real-Time" Fraud Detection

While this notebook demonstrates the model training, deploying it for real-time use would involve the following steps:

1.  **Save the Model:** After training, the `LGBMClassifier` model and the `StandardScaler` object would be saved to disk (e.g., using `joblib` or `pickle`).
2.  **Create an API:** A lightweight web server (e.g., using FastAPI or Flask) would load the saved model and scaler objects.
3.  **Expose an Endpoint:** The server would expose an API endpoint (e.g., `/predict_fraud`).
4.  **Real-Time Inference:** When a customer makes a transaction, the payment processing system would send the transaction's features (Amount, V1, V2, etc.) to this endpoint in a JSON payload. The API would then:
    a. Scale the incoming data using the loaded scaler.
    b. Feed the data to the model to get a fraud probability score.
    c. Return the score in milliseconds, allowing the system to instantly approve, block, or flag the transaction for manual review.
