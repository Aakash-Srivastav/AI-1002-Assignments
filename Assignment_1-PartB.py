import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# Load the iris dataset
iris = datasets.load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target labels: 0 = setosa, 1 = versicolor, 2 = virginica

# Convert the multiclass target into three binary classifications
y_setosa = (y == 0).astype(int)
y_versicolor = (y == 1).astype(int)
y_virginica = (y == 2).astype(int)

# Function to train and plot logistic regression for each feature

def plot_logistic_regression(X, y, feature_index, class_label):
    feature_values = X[:, feature_index].reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(feature_values, y, test_size=0.2, random_state=10)
    
    model = LogisticRegression(C=1.0)
    model.fit(X_train, y_train)
    
    # Predict probability values
    x_min, x_max = feature_values.min(), feature_values.max()
    x_range = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    y_prob = model.predict_proba(x_range)[:, 1]

    # Plot the data points with different markers
    plt.scatter(feature_values[y == 0], y[y == 0], marker='s', color='blue', label="Not " + class_label, alpha=0.6)
    plt.scatter(feature_values[y == 1], y[y == 1], marker='^', color='green', label=class_label, alpha=0.6)
    
    # Plot the logistic regression probability curve
    plt.plot(x_range, y_prob, 'g-', label='Logistic Regression (Positive Class)')
    plt.plot(x_range, 1 - y_prob, 'b--', label='Logistic Regression (Negative Class)')

    plt.xlabel(iris.feature_names[feature_index])
    plt.ylabel(f'Probability of {class_label}')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()

# Plot for each feature for setosa, versicolor, and virginica
for i in range(X.shape[1]):
    plot_logistic_regression(X, y_setosa, i, "Setosa")
    plot_logistic_regression(X, y_versicolor, i, "Versicolor")
    plot_logistic_regression(X, y_virginica, i, "Virginica")

# Function to train and plot decision boundaries for two features
def plot_decision_boundary(X, y_new, feature_indices, class_label):
    X_selected = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_new, test_size=0.2, random_state=10)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(C=1)
    model.fit(X_train, y_train)
    
    # Create a mesh grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    x0, x1 = np.meshgrid(
        np.linspace(x_min, x_max, 500).reshape(-1,1),
        np.linspace(y_min, y_max, 200).reshape(-1,1),
    )

    X_new = np.c_[x0.ravel(), x1.ravel()]
    Z = model.predict_proba(X_new)[:,1]

    # Boundary Line
    left_right = np.array([x_min, x_max])
    boundary = -(model.coef_[0][0] * left_right + model.intercept_[0])/model.coef_[0][1]

    # plt.figure(figsize=(6, 4))
    plt.contourf(x0, x1, Z.reshape(x0.shape), alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar(label='Probability')

    comparison = y_new == y
    equal_array = comparison.all()
    if equal_array != True:
        plt.plot(left_right ,boundary, "k--", linewidth = 3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='plasma',edgecolor='k')
    plt.xlabel(iris.feature_names[feature_indices[0]])
    plt.ylabel(iris.feature_names[feature_indices[1]])
    plt.title(f'Decision Boundary for {class_label}')
    plt.show()

# Train models using petal length & petal width and sepal length & sepal width
plot_decision_boundary(X, y_setosa, [2, 3], "Setosa")
plot_decision_boundary(X, y_versicolor, [2, 3], "Versicolor")
plot_decision_boundary(X, y_virginica, [2, 3], "Virginica")
plot_decision_boundary(X, y_setosa, [0, 1], "Setosa")
plot_decision_boundary(X, y_versicolor, [0, 1], "Versicolor")
plot_decision_boundary(X, y_virginica, [0, 1], "Virginica")

# Train a multinomial logistic regression model
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0)
X_selected = X[:, [2, 3]]  # Petal length and petal width
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=10)
model_multi.fit(X_train, y_train)

def multi_boundary(X, y, feature_indices, class_label):
    X_selected = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(C=1.0)
    model.fit(X_train, y_train)
    
    # Create a mesh grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_data).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00AA00'])

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.contour(xx, yy, Z, colors='black', linewidths=1)  # Explicit decision boundary
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=cmap_bold)
    
    plt.xlabel(iris.feature_names[feature_indices[0]])
    plt.ylabel(iris.feature_names[feature_indices[1]])
    plt.title(f'Decision Boundary for {class_label}')
    plt.show()

# Plot decision boundaries
multi_boundary(X, y, [2, 3], "Multiclass Classification")