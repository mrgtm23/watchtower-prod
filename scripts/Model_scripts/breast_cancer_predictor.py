import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train a Support Vector Machine
model = SVC(kernel='linear', probability=True, C=1.0)
model.fit(X, y)

# Save artifacts
joblib.dump(model, "breast_cancer_svm.pkl")
X.assign(target=y).to_csv("cancer_reference.csv", index=False)