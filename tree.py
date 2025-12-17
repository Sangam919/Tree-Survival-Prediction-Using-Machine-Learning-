import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Load the cleaned dataset
df = pd.read_csv("clean_tree_data1.csv")

# Separate features and target variable
X = df.drop("Target", axis=1)
y = df["Target"]

# Train-Test Split
# Stratified split is used to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
)

# Feature Scaling
# Standardization is applied to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Noise Addition
# Small Gaussian noise is added to reduce overfitting
# and improve model generalization
X_train += np.random.normal(0, 0.10, X_train.shape)
X_test  += np.random.normal(0, 0.10, X_test.shape)

# Decision Tree Model
# Model complexity is controlled to avoid overfitting
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    min_samples_split=40,
    min_samples_leaf=20,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Prediction and Evaluation
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))
print("Precision:", round(precision_score(y_test, pred), 2))
print("Recall:", round(recall_score(y_test, pred), 2))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
