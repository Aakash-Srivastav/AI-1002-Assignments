# README for AI-1002 Assignments
This repository contains implementations of various machine learning algorithms across five assignments. Below is a detailed description of each assignment.

## 1. Assignment_1-PartA: Linear and Polynomial Regression
### Description:
This script demonstrates linear and polynomial regression using synthetic datasets. It covers:

Generating synthetic data with noise.

Fitting linear regression models.

Visualizing the data and regression fits.

Extending to polynomial regression with varying degrees.

### Key Features:
Uses sklearn.linear_model.LinearRegression for model fitting.

Plots data points and regression lines using matplotlib.

Evaluates model coefficients and intercepts.

Compares polynomial fits of degrees 1, 2, 5, and 10.

## 2. Assignment_1-PartB: Logistic Regression for Classification
### Description:
This script focuses on logistic regression for binary and multiclass classification using the Iris dataset. It includes:

Binary classification for each Iris species (Setosa, Versicolor, Virginica).

Visualization of logistic regression probability curves for individual features.

Decision boundary plots for pairs of features.

Multiclass logistic regression using the multinomial approach.

### Key Features:
Uses sklearn.linear_model.LogisticRegression.

Plots probability curves and decision boundaries.

Standardizes features using StandardScaler.

Demonstrates multiclass classification with multi_class='multinomial'.

## 3. Assignment2_PartA: K-Nearest Neighbors (KNN) Classifier
### Description:
This script implements KNN classification on the Seeds and Iris datasets. It covers:

Data exploration and visualization.

Finding the optimal K value for KNN.

Plotting accuracy vs. K values for both datasets.

### Key Features:
Uses sklearn.neighbors.KNeighborsClassifier.

Splits data into training and testing sets.

Evaluates accuracy for K values from 1 to 25.

Generates scatter plots for feature-target relationships.

## 4. Assignment2_PartB: Naive Bayes Classifier
### Description:
This script demonstrates the Naive Bayes classifier for categorical data. It includes:

Encoding categorical features using LabelEncoder.

Training a CategoricalNB model with and without feature smoothing.

Calculating prior and conditional probabilities.

Predicting probabilities for given feature combinations.

### Key Features:
Handles small datasets with categorical features.

Displays prior and conditional probabilities in a readable format.

Answers specific probability questions (e.g., P(yes|high)).

## 5. Assignment3_main: Neural Networks with Different Activation Functions
### Description:
This script implements a neural network from scratch using various activation functions:

Sigmoid

Hyperbolic Tangent (tanh)

Rectified Linear Unit (ReLU)

Leaky ReLU

### Key Features:
Custom implementation of forward and backward propagation.

Training loop with 200,000 epochs.

Evaluation of final weights, errors, and accuracy.

Prediction on new data for each activation function.

### Usage:
Run the script to train the neural network and observe the performance of each activation function. The script outputs:

Final weights for each layer.

Prediction errors.

Accuracy of the model.

Predictions for custom input data.
