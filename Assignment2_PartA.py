import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load the wheat kernel dataset
df = pd.read_csv(rf'D:\VS_Code\AI_SEM1\AI_Algorithms\assignments\Assignment_2\seeds.csv')

# Describe the dataset
print("\n4. Dataset Description:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nTarget value counts:")
print(df['Target'].value_counts())

print("\n5. Generating scatter plots for independent features...")
features = df.columns[:-1]
target = df['Target']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.scatter(df[feature], target, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Target')
    plt.title(f'{feature} vs Target')
plt.tight_layout()
plt.show()

# Prepare data for KNN
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Find optimal K value
print("\n1. Finding optimal K value...")
k_values = range(1, 26)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

optimal_k = k_values[np.argmax(accuracy_scores)]
print(f"Optimal K value: {optimal_k}")

# Plot accuracy vs K values
print("\n2. Plotting accuracy vs K values...")
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Value of K')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Different Values of K (Seeds Dataset)')
plt.xticks(range(0, 26, 5))
plt.grid(True)
plt.show()

iris = load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'],iris['target']], columns=iris['feature_names']+['target'])
iris.head()

#X variable, flowers' feature
#y variable, target values
X = iris.drop(['target'], axis=1)
y = iris['target']

X_train_iris,X_test_iris,y_train_iris,y_test_iris = train_test_split(X,y,test_size = 0.4,random_state= 0)
k_range = list(range(1,25))
scores_iris = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_iris, y_train_iris)
    y_pred = knn.predict(X_test_iris)
    scores_iris.append(metrics.accuracy_score(y_test_iris, y_pred))
plt.plot(k_range, scores_iris)
plt.xlabel('Value of K')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for different values of k (Iris Dataset)')
plt.show()