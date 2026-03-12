import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Regressor
model = HistGradientBoostingRegressor(max_iter=100)
model.fit(X, y)

# Save artifacts
joblib.dump(model, "housing_gb.pkl")
X.assign(target=y).to_csv("housing_reference.csv", index=False)