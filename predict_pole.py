import joblib
import pandas as pd

# Load trained model
model = joblib.load("f1_quali_model.pkl")

print("=== F1 QUALIFYING POLE PREDICTOR ===")

# Take user input
driver_avg = float(input("Driver average qualifying position: "))
team_avg = float(input("Team average qualifying position: "))
recent_form = float(input("Driver recent form (last 5 races avg): "))

# Create dataframe
input_data = pd.DataFrame([[driver_avg, team_avg, recent_form]],
                          columns=['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm'])

# Predict
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print("\nPrediction Result:")
if prediction == 1:
    print(f"🔥 High chance of POLE! Probability: {probability:.2f}")
else:
    print(f"Pole unlikely. Probability: {probability:.2f}")
