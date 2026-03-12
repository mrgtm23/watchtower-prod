import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "iris_rf.pkl")

# Save reference dataset
X.assign(target=y).to_csv("iris_reference.csv", index=False)

print("Iris model and reference dataset saved!")