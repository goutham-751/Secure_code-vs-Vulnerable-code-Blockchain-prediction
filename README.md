# Secure_code-vs-Vulnerable-code-Blockchain-prediction
Solidity Smart Contract Fraud Detection

Overview

This project implements a fraud detection model using machine learning to classify Solidity smart contracts as either Secure or Vulnerable. The model is trained on hashed contract IDs extracted from datasets containing secure and vulnerable contracts.

Features

Preprocessing of Solidity contract hashes.

Feature extraction using TF-IDF Vectorization.

Training a Random Forest Classifier to detect fraudulent contracts.

Interactive CLI to check contract security status.

Model evaluation with accuracy and classification reports.

Dataset

The model is trained on:

BCCC-VolSCs-2023_Secure.csv - Secure contracts

BCCC-VolSCs-2023_Vulnerable.csv - Vulnerable contracts

Installation

Prerequisites

Python 3.x

Required libraries: numpy, pandas, sklearn, re

Setup

Clone this repository:

git clone https://github.com/yourusername/solidity-fraud-detection.git
cd solidity-fraud-detection

Install dependencies:

pip install -r requirements.txt

Usage

Place the dataset files inside the project directory.

Run the fraud detection script:

python solidity_prediction.py

Enter Solidity contract hash IDs to check if they are Secure or Vulnerable.

Type exit to close the interactive session.

Model Performance

The trained Random Forest Classifier provides high accuracy in classifying secure and fraudulent contracts.

Performance metrics include accuracy score and classification report.

Future Enhancements

Improve feature engineering for better contract classification.

Extend the model to analyze full contract code instead of just hash IDs.

Deploy as a web-based API for real-time analysis.

License

This project is licensed under the MIT License.

Author: Goutham Kumar

