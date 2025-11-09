
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

Path("models").mkdir(exist_ok=True)

df = pd.read_csv("data/synthetic.csv")
X = df[["frequency", "amplitude"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("RandomForest classification report:\n", classification_report(y_test, y_pred))
joblib.dump(rf, "models/rf_model.joblib")
print("Saved RF model to models/rf_model.joblib")

# SVM (optional)
svm = SVC(kernel='rbf', probability=True, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred2 = svm.predict(X_test)
print("SVM classification report:\n", classification_report(y_test, y_pred2))
joblib.dump(svm, "models/svm_model.joblib")
print("Saved SVM model to models/svm_model.joblib")
