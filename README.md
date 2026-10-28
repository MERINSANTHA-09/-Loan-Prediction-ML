# LoanAI — Loan Prediction using Machine Learning

## Project Overview

LoanAI is a Machine Learning web application built using Python and Streamlit. It analyses loan application data and predicts loan approval status using a Random Forest Classifier.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Features

- Interactive Loan Prediction Dashboard
- CSV dataset upload and preview
- Automatic data preprocessing
- Missing-value handling and categorical encoding
- Random Forest classification model
- Model accuracy and train-test statistics
- Loan approval and rejection predictions
- Model confidence display
- Dataset insights and missing-value analysis
- Confusion matrix and feature importance visualization
- Download prediction history as CSV
- Download cleaned dataset as CSV
- Custom dark-themed user interface

## Project Files

- `app.py` — Main Streamlit application
- `Loan Prediction Dataset.csv` — Sample dataset used by the application

## Installation and Setup

### 1. Install Python

Install Python on your computer if it is not already installed.

### 2. Install Required Libraries

Open the terminal in the project folder and run:

```bash
pip install streamlit pandas numpy scikit-learn
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Number of Trees: 200
- Training Data: 80%
- Testing Data: 20%
- Preprocessing: Missing-value imputation and one-hot encoding

Model accuracy is evaluated using a held-out test dataset. Actual loan approval decisions may differ from model predictions.

## Author

Merin Santha

## Disclaimer

This project is intended for educational purposes only. Predictions are not official bank decisions or financial advice.