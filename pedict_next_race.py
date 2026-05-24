import pandas as pd
import joblib

# Load trained model
model = joblib.load("f1_grid_model.pkl")

# Load dataset
df = pd.read_csv("features_dataset.csv")

# -----------------------------
# Get latest stats per driver
# -----------------------------
df = df.sort_values(['FullName', 'Year'])
latest = df.groupby('FullName').tail(1).copy()

# -----------------------------
# OPTIONAL: Track difficulty adjustment
# (0 = normal, -1 = favors top teams, +1 = chaotic track)
# Example: Monaco = -0.5 (hard to overtake)
# -----------------------------
track_factor = float(input("Enter track factor (-1 to +1): "))

latest['RecentForm'] = latest['RecentForm'] + track_factor

# -----------------------------
# Predict
# -----------------------------
features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']
latest['RawPrediction'] = model.predict(latest[features])

# Convert to grid positions
latest['GridPosition'] = latest['RawPrediction'].rank(method='first').astype(int)

# Final grid
grid = latest[['FullName', 'TeamName', 'GridPosition']].sort_values('GridPosition')

print("\n🏁 PREDICTED GRID FOR NEXT RACE")
print(grid.to_string(index=False))
