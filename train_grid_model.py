import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("features_dataset.csv")

# Features
features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']
X = df[features]

# Target = actual qualifying position
y = df['Position']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=300, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, preds)
print("Mean Absolute Error:", mae)

# Save model
joblib.dump(model, "f1_grid_model.pkl")
print("Grid prediction model saved!")
