import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set random seed
np.random.seed(10)

def lr_lin():

    # Part 1: y = 12x - 4 + noise
    x = np.random.rand(200, 1)  # Sample from uniform distribution
    noise = np.random.randn(200, 1)  # Gaussian noise
    y = 12 * x - 4 + noise

    # Plot the data
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Data: y = 12x - 4 + noise")
    plt.show()

    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict on test set
    y_pred = model.predict(x_test)

    print(f'coef_ attribute of trained model: {model.coef_}')
    print(f'intercept_ attribute of trained model: {model.intercept_}')

    # Plot predictions
    plt.scatter(x_test, y_test, color='blue', label='Ground Truth')
    plt.plot(x_test, y_pred, color='red', label='Predictions')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

    # Print model parameters
    print(f"Model Coefficient: {model.coef_[0][0]}")
    print(f"Model Intercept: {model.intercept_[0]}")

lr_lin()

def lr_poly():

    # Part 2: y = 0.75x + 2x^2 + 1
    x = np.random.uniform(-3, 3, (200, 1))
    y = 0.75 * x + 2 * x**2 + 1 + np.random.randn(200, 1)  # Quadratic function with noise

    # Plot the data
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Data: y = 0.75x + 2x^2 + 1 + noise")
    plt.show()

    # Split into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict on test set
    y_pred = model.predict(x_test)

    # Plot predictions
    plt.scatter(x_test, y_test, color='blue', label='Ground Truth')
    plt.plot(x_test, y_pred, color='red', label='Predictions')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

    # Polynomial Regression
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly_train = poly_features.fit_transform(x_train)
    x_poly_test = poly_features.transform(x_test)

    # Fit Polynomial Regression Model
    poly_model = LinearRegression()
    poly_model.fit(x_poly_train, y_train)

    # Predict with Polynomial Model
    y_poly_pred = poly_model.predict(x_poly_test)

    # Generate smooth predictions
    x_smooth = np.linspace(-3, 3, 100).reshape(-1, 1)
    x_smooth_poly = poly_features.transform(x_smooth)
    y_smooth_pred = poly_model.predict(x_smooth_poly)

    # Plot predictions
    plt.scatter(x_test, y_test, color='blue', label='Ground Truth')
    plt.plot(x_smooth, y_smooth_pred, color='red', label='Polynomial Predictions')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.show()

    # Train models with varying degrees
    for degree in [1, 2, 5, 10]:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)
        x_poly_test = poly_features.transform(x_test)
        x_smooth_poly = poly_features.transform(x_smooth)
        
        poly_model = LinearRegression()
        poly_model.fit(x_poly_train, y_train)
        y_poly_smooth_pred = poly_model.predict(x_smooth_poly)
        
        plt.scatter(x_test, y_test, color='blue', alpha=0.5, label='Ground Truth')
        plt.plot(x_smooth, y_poly_smooth_pred, label=f'Poly Degree {degree}', linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Polynomial Regression Fit (Degree {degree})")
        plt.legend()
        plt.show()

lr_poly()

a = []