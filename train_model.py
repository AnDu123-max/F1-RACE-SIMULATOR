import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load feature dataset
df = pd.read_csv("features_dataset.csv")

# Select features
features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']
X = df[features]

# Target (Pole Position)
y = df['PolePosition']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "f1_quali_model.pkl")

print("Model saved as f1_quali_model.pkl")
