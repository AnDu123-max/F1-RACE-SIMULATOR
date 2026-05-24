import pandas as pd
import joblib

# Load model
model = joblib.load("f1_grid_model.pkl")

# Load dataset
df = pd.read_csv("features_dataset.csv")

# Get latest stats per driver
df = df.sort_values(['FullName', 'Year'])
latest_data = df.groupby('FullName').tail(1)

features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']

# Predict raw values
latest_data['RawPrediction'] = model.predict(latest_data[features])

# Rank predictions (smallest = best position)
latest_data['GridPosition'] = latest_data['RawPrediction']\
    .rank(method='first').astype(int)

# Sort final grid
grid = latest_data[['FullName', 'TeamName', 'GridPosition']]\
    .sort_values('GridPosition')

print("\n🏁 AI PREDICTED QUALIFYING GRID")
print(grid.to_string(index=False))
