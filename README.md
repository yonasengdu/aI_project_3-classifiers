# aI_project_3-classifiers
# Machine Learning Classifier Implementations

This repository contains Python implementations of Naive Bayes and Logistic Regression classifiers for three different datasets: MNIST Digit Dataset, BBC Text Classification, and Demo Weather Dataset. The implementations include hyperparameter tuning and feature extraction techniques.

## Features

- Modular structure with separate classes for data loading, preprocessing, feature extraction, classifiers, evaluation, and plotting.
- Supports loading and preprocessing of MNIST, BBC, and Demo Weather datasets.
- Provides multiple feature extraction methods for each dataset.
- Implements Naive Bayes and Logistic Regression classifiers.
- Performs hyperparameter tuning by testing different smoothing values for Naive Bayes and learning rates for Logistic Regression.
- Calculates accuracy of the classifiers and plots the accuracy change based on hyperparameter variations.

## Dataset Details

1. MNIST Digit Dataset: A popular dataset of handwritten digits (0-9). The goal is to classify the images correctly.

2. BBC Text Classification: A dataset of news articles from different categories (business, entertainment, politics, sport, tech). The task is to classify the articles into their respective categories.

3. Demo Weather Dataset: A synthetic dataset generated to demonstrate weather classification. It includes features such as temperature, humidity, wind, and outlook, and the task is to predict whether it is "hot" or "not hot" based on these features.

## Usage

1. Clone the repository:

git clone https://github.com/your-username/machine-learning-classifiers.git

2. Install the required dependencies:

pip install mnist # for MNIST dataset loading


3. Update the dataset paths in the code:

- For MNIST dataset: Update the `MNISTDataLoader` class with the correct dataset directory path.
- For BBC dataset: Update the `BBCDataLoader` class with the correct dataset file path.
- For Demo Weather dataset: No path update required as it is generated within the code.

4. Run the main script:

python main.py


The script will load the datasets, perform hyperparameter tuning, train and test the classifiers, and plot the accuracy change based on hyperparameter variations.

## Contributions

Contributions to this project are welcome! If you have any improvements or new features to add, please feel free to open an issue or submit a pull request.

