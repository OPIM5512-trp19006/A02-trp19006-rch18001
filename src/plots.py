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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
		learning_rate_init=0.001,
	),
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
mlp = model[-1]

# Train a second model with a higher learning rate.
tuned_model = make_pipeline(
	StandardScaler(),
	MLPRegressor(
		hidden_layer_sizes=(100,),
		early_stopping=True,
		validation_fraction=0.1,
		n_iter_no_change=10,
		random_state=42,
		max_iter=500,
		learning_rate_init=0.01,
	),
)
tuned_model.fit(X_train, y_train)
y_train_tuned_pred = tuned_model.predict(X_train)
y_tuned_pred = tuned_model.predict(X_test)
tuned_mlp = tuned_model[-1]

def print_metrics(name, actual_train, predicted_train, actual_test, predicted_test, iterations):
	print(f"\n{name} (training iterations: {iterations})")
	for split, actual, predicted in (
		("Train", actual_train, predicted_train),
		("Test", actual_test, predicted_test),
	):
		print(
			f"{split}: R^2={r2_score(actual, predicted):.4f}, "
			f"MAE={mean_absolute_error(actual, predicted):.4f}, "
			f"RMSE={mean_squared_error(actual, predicted) ** 0.5:.4f}"
		)

print_metrics("Baseline model (learning rate 0.001)", y_train, y_train_pred, y_test, y_pred, mlp.n_iter_)
print_metrics(
	"Tuned model (learning rate 0.01)",
	y_train,
	y_train_tuned_pred,
	y_test,
	y_tuned_pred,
	tuned_mlp.n_iter_,
)

# Plot actual versus predicted values for the training set.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_train, y_train_pred, alpha=0.25, s=12)
plot_min = min(y_train.min(), y_train_pred.min())
plot_max = max(y_train.max(), y_train_pred.max())
ax.plot([plot_min, plot_max], [plot_min, plot_max], color="black", linestyle="--")
ax.set_xlabel("Actual MedHouseVal")
ax.set_ylabel("Predicted MedHouseVal")
ax.set_title("Training Set: Actual vs Predicted")
ax.grid(alpha=0.2)
fig.tight_layout()

figs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")
os.makedirs(figs_dir, exist_ok=True)
fig.savefig(os.path.join(figs_dir, "train_actual_vs_pred.png"), dpi=300)
plt.close(fig)

# Plot actual versus predicted values for the test set.
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.25, s=12)
plot_min = min(y_test.min(), y_pred.min())
plot_max = max(y_test.max(), y_pred.max())
ax.plot([plot_min, plot_max], [plot_min, plot_max], color="black", linestyle="--")
ax.set_xlabel("Actual MedHouseVal")
ax.set_ylabel("Predicted MedHouseVal")
ax.set_title("Test Set: Actual vs Predicted")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(figs_dir, "test_actual_vs_pred.png"), dpi=300)
plt.close(fig)

