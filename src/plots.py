from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Define features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Train the MLP on scaled features and stop when the validation score stops improving.
model = make_pipeline(
	StandardScaler(),
	MLPRegressor(
		hidden_layer_sizes=(100,),
		early_stopping=True,
		validation_fraction=0.1,
		n_iter_no_change=10,
		random_state=42,
		max_iter=500,
	),
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mlp = model[-1]
print(f"MLP test R^2: {r2_score(y_test, y_pred):.4f}")
print(f"Training iterations: {mlp.n_iter_}")


