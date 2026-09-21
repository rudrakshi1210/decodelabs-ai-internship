# DecodeLabs AI Internship - Project 2: Data Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# 1. Load the Iris Benchmark Dataset
iris = load_iris()
X = iris.data  # 4 features: sepal length/width, petal length/width
y = iris.target  # 3 classes: Setosa, Versicolor, Virginica

print("=" * 55)
print(" DecodeLabs AI - Supervised Learning Pipeline ")
print("=" * 55)
print(f"Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
print(f"Target classes: {list(iris.target_names)}\n")

# 2. Train-Test Split (80% Train, 20% Test) with shuffling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# 3. Feature Scaling (StandardScaler: Mean = 0, Variance = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Instantiate and Train KNN Model (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 5. Predict on Unseen Test Data
y_pred = knn.predict(X_test_scaled)

# 6. Model Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {acc * 100:.2f}%")
print(f"Weighted F1 Score: {f1:.4f}\n")

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))