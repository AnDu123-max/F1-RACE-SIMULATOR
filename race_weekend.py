import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ---------------- DRIVER LIST (22) ----------------
drivers = [
    "Verstappen","Perez","Hamilton","Russell","Leclerc","Sainz",
    "Norris","Piastri","Alonso","Stroll","Gasly","Ocon",
    "Albon","Sargeant","Tsunoda","Ricciardo",
    "Bottas","Zhou","Magnussen","Hulkenberg",
    "Schumacher","Pourchaire"
]

# ---------------- DRIVER ATTRIBUTES ----------------
driver_skill = {d: random.randint(75, 97) for d in drivers}
experience = {d: random.randint(60, 95) for d in drivers}
aggression = {d: random.randint(60, 95) for d in drivers}

# ---------------- TRACKS (24) ----------------
tracks = [
    "Bahrain","Jeddah","Australia","Suzuka","China","Miami",
    "Imola","Monaco","Canada","Spain","Austria","Silverstone",
    "Hungary","Belgium","Netherlands","Monza","Singapore",
    "USA","Mexico","Brazil","Vegas","Qatar","Abu Dhabi","Germany"
]

# ---------------- TRAINING DATA GENERATION ----------------
data = []

for _ in range(4000):  # historical simulated races
    for d in drivers:
        track_factor = random.uniform(0.8, 1.2)
        chaos = random.uniform(-5, 5)
        performance = (driver_skill[d]*0.5 +
                       experience[d]*0.3 +
                       aggression[d]*0.2) * track_factor + chaos

        data.append([
            driver_skill[d],
            experience[d],
            aggression[d],
            track_factor,
            chaos,
            performance
        ])

df = pd.DataFrame(data, columns=[
    "skill","experience","aggression","track_factor","chaos","performance"
])

X = df.drop("performance", axis=1)
y = df["performance"]

# ---------------- TRAIN ML MODEL ----------------
model = RandomForestRegressor(n_estimators=150)
model.fit(X, y)

print("✅ ML Model Trained Successfully!\n")

# ---------------- F1 POINT SYSTEM ----------------
points_table = [25,18,15,12,10,8,6,4,2,1]
championship = {d:0 for d in drivers}

# ---------------- RACE SIMULATION ----------------
def simulate_race(track):
    print(f"\n🏁 {track} Grand Prix")

    race_data = []
    for d in drivers:
        track_factor = random.uniform(0.8,1.2)
        chaos = random.uniform(-6,6)

        features = [[
            driver_skill[d],
            experience[d],
            aggression[d],
            track_factor,
            chaos
        ]]

        predicted_perf = model.predict(features)[0]
        race_data.append((d, predicted_perf))

    results = sorted(race_data, key=lambda x: x[1], reverse=True)

    for pos, (driver, _) in enumerate(results,1):
        print(f"{pos:2}. {driver}")

    return [r[0] for r in results]

# ---------------- FULL SEASON ----------------
for track in tracks:
    finish_order = simulate_race(track)

    for pos, driver in enumerate(finish_order[:10]):
        championship[driver] += points_table[pos]

# ---------------- FINAL STANDINGS ----------------
print("\n🏆 2026 WORLD CHAMPIONSHIP 🏆\n")
standings = sorted(championship.items(), key=lambda x: x[1], reverse=True)

for pos, (driver, pts) in enumerate(standings,1):
    print(f"{pos:2}. {driver:15} {pts} pts")

print("\n🌟 WORLD CHAMPION:", standings[0][0])
