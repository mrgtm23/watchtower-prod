import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

# Load data
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model with specific hyperparams to test CT retrieval
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)

# Save artifacts
joblib.dump(model, "wine_quality_rf.pkl")
X.assign(target=y).to_csv("wine_reference.csv", index=False)