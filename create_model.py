# create_model.py
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Create a dummy model (simple logistic regression)
print("Creating dummy model...")
# Dummy data: 4 features, 2 classes
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# 2. Save the model to disk
model_filename = "simple_iris_classifier.pkl"
joblib.dump(model, model_filename)

print(f"✅ Dummy model created: {model_filename}")
print("You can now upload this file to the API.")