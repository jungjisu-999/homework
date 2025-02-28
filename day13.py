import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

#data loading
mpg = sns.load_dataset('mpg')
mpg.dropna(inplace=True)

# Set Target and Independent Variables
target = 'mpg'
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
X = mpg[features]
y = mpg[target]

# Categorical data conversion
ct = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), ['origin'])],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)

# data segmentation
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Learning
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Performing predictions
y_pred = model.predict(X_test_scaled)

# Performance Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()
