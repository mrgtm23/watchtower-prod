import pandas as pd
import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = load_diabetes(as_frame=True)

X = data.data
y = data.target

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

# Save trained model
joblib.dump(model, "diabetes_rf.pkl")

# Save dataset
X.to_csv("diabetes_dataset.csv", index=False)

# Save reference dataset (with target)
reference = X.copy()
reference["target"] = y
reference.to_csv("diabetes_reference.csv", index=False)

print("Model trained and files generated successfully!")