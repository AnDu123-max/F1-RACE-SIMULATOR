import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize app
app = FastAPI(title="F1 2026 API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    grid_model = joblib.load(os.path.join(MODELS_DIR, "f1_grid_model.pkl"))
    dataset_df = pd.read_csv(os.path.join(DATA_DIR, "features_dataset.csv"))
except Exception as e:
    print(f"Error loading models or dataset: {e}")
    grid_model = None
    dataset_df = None

# Mock F1 2026 Teams
TEAMS_DATA = [
    {"name": "Red Bull Racing", "drivers": ["Max Verstappen", "Liam Lawson"], "engine": "Red Bull Powertrains"},
    {"name": "Mercedes", "drivers": ["George Russell", "Andrea Kimi Antonelli"], "engine": "Mercedes"},
    {"name": "Ferrari", "drivers": ["Charles Leclerc", "Lewis Hamilton"], "engine": "Ferrari"},
    {"name": "McLaren", "drivers": ["Lando Norris", "Oscar Piastri"], "engine": "Mercedes"},
    {"name": "Aston Martin", "drivers": ["Fernando Alonso", "Lance Stroll"], "engine": "Honda"},
    {"name": "Alpine", "drivers": ["Pierre Gasly", "Jack Doohan"], "engine": "Renault"},
    {"name": "Williams", "drivers": ["Alexander Albon", "Carlos Sainz"], "engine": "Mercedes"},
    {"name": "Racing Bulls", "drivers": ["Yuki Tsunoda", "Isack Hadjar"], "engine": "Red Bull Powertrains"},
    {"name": "Audi", "drivers": ["Nico Hulkenberg", "Gabriel Bortoleto"], "engine": "Ferrari"},
    {"name": "Haas F1 Team", "drivers": ["Esteban Ocon", "Oliver Bearman"], "engine": "Ferrari"},
    {"name": "Cadillac", "drivers": ["Valtteri Bottas", "Sergio Perez"], "engine": "Ferrari"}
]

# Driver Ratings
DRIVER_RATINGS = {
    "Max Verstappen": {"Pace": 98, "Racecraft": 97, "Experience": 85},
    "Liam Lawson": {"Pace": 83, "Racecraft": 80, "Experience": 55},
    "George Russell": {"Pace": 91, "Racecraft": 88, "Experience": 75},
    "Andrea Kimi Antonelli": {"Pace": 85, "Racecraft": 78, "Experience": 40},
    "Charles Leclerc": {"Pace": 95, "Racecraft": 90, "Experience": 78},
    "Lewis Hamilton": {"Pace": 93, "Racecraft": 96, "Experience": 99},
    "Lando Norris": {"Pace": 93, "Racecraft": 89, "Experience": 76},
    "Oscar Piastri": {"Pace": 90, "Racecraft": 86, "Experience": 60},
    "Fernando Alonso": {"Pace": 89, "Racecraft": 95, "Experience": 99},
    "Lance Stroll": {"Pace": 81, "Racecraft": 82, "Experience": 80},
    "Pierre Gasly": {"Pace": 84, "Racecraft": 83, "Experience": 78},
    "Jack Doohan": {"Pace": 79, "Racecraft": 77, "Experience": 45},
    "Alexander Albon": {"Pace": 86, "Racecraft": 85, "Experience": 75},
    "Carlos Sainz": {"Pace": 90, "Racecraft": 91, "Experience": 82},
    "Yuki Tsunoda": {"Pace": 84, "Racecraft": 82, "Experience": 70},
    "Isack Hadjar": {"Pace": 78, "Racecraft": 75, "Experience": 40},
    "Nico Hulkenberg": {"Pace": 85, "Racecraft": 84, "Experience": 90},
    "Gabriel Bortoleto": {"Pace": 80, "Racecraft": 78, "Experience": 45},
    "Esteban Ocon": {"Pace": 84, "Racecraft": 85, "Experience": 77},
    "Oliver Bearman": {"Pace": 81, "Racecraft": 80, "Experience": 50},
    "Valtteri Bottas": {"Pace": 83, "Racecraft": 85, "Experience": 92},
    "Sergio Perez": {"Pace": 85, "Racecraft": 86, "Experience": 90}
}

# Driver Numbers
DRIVER_NUMBERS = {
    "Max Verstappen": 1, "Sergio Perez": 11, "Lewis Hamilton": 44, "George Russell": 63,
    "Charles Leclerc": 16, "Carlos Sainz": 55, "Lando Norris": 4, "Oscar Piastri": 81,
    "Fernando Alonso": 14, "Lance Stroll": 18, "Pierre Gasly": 10, "Esteban Ocon": 31,
    "Alexander Albon": 23, "Yuki Tsunoda": 22, "Valtteri Bottas": 77, "Guanyu Zhou": 24,
    "Nico Hulkenberg": 27, "Kevin Magnussen": 20, "Liam Lawson": 30, "Andrea Kimi Antonelli": 12,
    "Jack Doohan": 7, "Isack Hadjar": 17, "Gabriel Bortoleto": 85, "Oliver Bearman": 87
}

# Mini RAG KB
F1_KB = [
    "F1 2026 introduces major aerodynamic changes and 100% sustainable fuels.",
    "Max Verstappen drives for Red Bull Racing and is a multiple-time world champion.",
    "Ferrari is known for their iconic red livery and powerful engines.",
    "The 2026 engine regulations feature roughly 50% internal combustion and 50% electrical power.",
    "McLaren showed massive improvement in recent years, spearheaded by Lando Norris and Oscar Piastri.",
    "Mercedes is working to return to their dominant form with George Russell and Lewis Hamilton.",
]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(F1_KB)

class ChatQuery(BaseModel):
    query: str

@app.get("/api/teams")
def get_teams():
    enriched_teams = []
    for team in TEAMS_DATA:
        t = team.copy()
        t["driver_ratings"] = {d: DRIVER_RATINGS.get(d, {"Pace": 80, "Racecraft": 80, "Experience": 50}) for d in team["drivers"]}
        enriched_teams.append(t)
    return enriched_teams

@app.get("/api/predict/qualifying")
def predict_qualifying(track: Optional[str] = None):
    if dataset_df is None or grid_model is None:
        raise HTTPException(status_code=500, detail="Model or dataset not loaded")
    
    # Logic from predict_grid.py
    df = dataset_df.sort_values(['FullName', 'Year'])
    latest_data = df.groupby('FullName').tail(1).copy()
    
    grid_rows = []
    for team in TEAMS_DATA:
        for driver in team["drivers"]:
            if driver in latest_data['FullName'].values:
                row = latest_data[latest_data['FullName'] == driver].copy()
                row['TeamName'] = team["name"]
            else:
                row = pd.DataFrame([{
                    'FullName': driver, 'TeamName': team["name"],
                    'DriverAvgQualiPos': 15.0, 'TeamAvgQualiPos': 10.0, 'RecentForm': 0.5
                }])
            grid_rows.append(row)
            
    current_grid_df = pd.concat(grid_rows, ignore_index=True)
    features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']
    current_grid_df[features] = current_grid_df[features].fillna(10.0)
    
    import numpy as np
    import hashlib
    np.random.seed(int(hashlib.sha256((track or "None").encode('utf-8')).hexdigest(), 16) % 10**8)
    
    weather_options = [
        {"condition": "Sunny", "icon": "☀️", "temp": f"{np.random.randint(25, 35)}°C"},
        {"condition": "Cloudy", "icon": "☁️", "temp": f"{np.random.randint(18, 26)}°C"},
        {"condition": "Rain", "icon": "🌧️", "temp": f"{np.random.randint(15, 22)}°C"},
        {"condition": "Mixed", "icon": "🌦️", "temp": f"{np.random.randint(18, 25)}°C"}
    ]
    weather = weather_options[np.random.randint(0, len(weather_options))]
    
    raw_pred = grid_model.predict(current_grid_df[features])
    noise_level = 1.0 if weather["condition"] in ["Rain", "Mixed"] else 0.5
    noise = np.random.normal(0, noise_level, size=len(raw_pred))
    current_grid_df['RawPrediction'] = raw_pred + noise
    
    current_grid_df['GridPosition'] = current_grid_df['RawPrediction'].rank(method='first').astype(int)
    
    def generate_q_times(pos):
        base_time = 92.0 # 1:32.000
        q1_time = base_time + 1.2 + np.random.uniform(0.1, 1.5) + (pos * 0.15)
        q2_time = base_time + 0.5 + np.random.uniform(0.1, 1.0) + (pos * 0.1) if pos <= 15 else None
        q3_time = base_time + np.random.uniform(0.0, 0.5) + (pos * 0.08) if pos <= 10 else None
        
        def format_time(t):
            if t is None: return ""
            m = int(t // 60)
            s = t % 60
            return f"{m}:{s:06.3f}"
        
        return format_time(q1_time), format_time(q2_time), format_time(q3_time)

    grid_records = []
    for _, row in current_grid_df.sort_values('GridPosition').iterrows():
        pos = int(row['GridPosition'])
        q1, q2, q3 = generate_q_times(pos)
        laps = int(np.random.randint(15, 22) if pos <= 10 else (np.random.randint(12, 18) if pos <= 15 else np.random.randint(8, 12)))
        
        grid_records.append({
            "GridPosition": pos,
            "FullName": row['FullName'],
            "TeamName": row['TeamName'],
            "DriverNumber": DRIVER_NUMBERS.get(row['FullName'], int(np.random.randint(2, 99))),
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Laps": laps
        })
    
    return {"weather": weather, "grid": grid_records}

@app.get("/api/predict/race")
def predict_race(track: Optional[str] = None):
    if dataset_df is None or grid_model is None:
        raise HTTPException(status_code=500, detail="Model or dataset not loaded")
    
    df = dataset_df.sort_values(['FullName', 'Year'])
    latest_data = df.groupby('FullName').tail(1).copy()
    
    grid_rows = []
    for team in TEAMS_DATA:
        for driver in team["drivers"]:
            if driver in latest_data['FullName'].values:
                row = latest_data[latest_data['FullName'] == driver].copy()
                row['TeamName'] = team["name"]
            else:
                row = pd.DataFrame([{
                    'FullName': driver, 'TeamName': team["name"],
                    'DriverAvgQualiPos': 15.0, 'TeamAvgQualiPos': 10.0, 'RecentForm': 0.5
                }])
            grid_rows.append(row)
            
    current_grid_df = pd.concat(grid_rows, ignore_index=True)
    features = ['DriverAvgQualiPos', 'TeamAvgQualiPos', 'RecentForm']
    current_grid_df[features] = current_grid_df[features].fillna(10.0)
    
    raw_pred = grid_model.predict(current_grid_df[features])
    
    # Add random noise to simulate race variation
    import numpy as np
    import hashlib
    np.random.seed(int(hashlib.sha256((f"race_{track}" or "None").encode('utf-8')).hexdigest(), 16) % 10**8)
    
    weather_options = [
        {"condition": "Sunny", "icon": "☀️", "temp": f"{np.random.randint(25, 35)}°C"},
        {"condition": "Cloudy", "icon": "☁️", "temp": f"{np.random.randint(18, 26)}°C"},
        {"condition": "Rain", "icon": "🌧️", "temp": f"{np.random.randint(15, 22)}°C"},
        {"condition": "Mixed", "icon": "🌦️", "temp": f"{np.random.randint(18, 25)}°C"}
    ]
    weather = weather_options[np.random.randint(0, len(weather_options))]
    
    noise_level = 2.5 if weather["condition"] in ["Rain", "Mixed"] else 1.5
    race_perf = raw_pred + np.random.normal(0, noise_level, size=len(raw_pred))
    
    current_grid_df['RacePosition'] = pd.Series(race_perf, index=current_grid_df.index).rank(method='first').astype(int)
    
    # Calculate win probabilities based on performance score (lower is better)
    # Using softmax on negated performance scores
    scores = -1.0 * race_perf
    exp_scores = np.exp(scores - np.max(scores)) # subtract max for numerical stability
    probs = exp_scores / np.sum(exp_scores)
    current_grid_df['WinProbability'] = (probs * 100).round(1)
    
    total_laps_race = int(50 + np.random.randint(0, 20))
    points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
    
    grid_records = []
    base_time = total_laps_race * 90.0 # ~1:30 per lap
    winner_time = base_time + np.random.uniform(10, 30)
    
    for _, row in current_grid_df.sort_values('RacePosition').iterrows():
        pos = int(row['RacePosition'])
        pts = points_map.get(pos, 0)
        
        if pos == 1:
            m = int(winner_time // 60)
            s = winner_time % 60
            h = m // 60
            m = m % 60
            time_str = f"{h}:{m:02d}:{s:06.3f}"
        elif pos > 18:
            time_str = "DNF"
            pts = 0
        else:
            gap = np.random.uniform(2.0, 8.0) * pos + np.random.uniform(0, 5)
            time_str = f"+{gap:.3f}s"
            
        laps = total_laps_race if pos <= 18 else int(total_laps_race - np.random.randint(1, 20))
        
        grid_records.append({
            "RacePosition": pos,
            "FullName": row['FullName'],
            "TeamName": row['TeamName'],
            "DriverNumber": DRIVER_NUMBERS.get(row['FullName'], int(np.random.randint(2, 99))),
            "TimeRetired": time_str,
            "Laps": laps,
            "Points": pts,
            "WinProbability": row['WinProbability']
        })
    
    # Generate mock tire strategy
    strategies = [
        {"name": "One Stop (Soft -> Hard)", "pit_window": "Lap 18-24", "compounds": ["Soft", "Hard"], "prob": 65},
        {"name": "Two Stop (Medium -> Hard -> Soft)", "pit_window": "Lap 15-20, 38-45", "compounds": ["Medium", "Hard", "Soft"], "prob": 25},
        {"name": "Alternative (Hard -> Medium)", "pit_window": "Lap 30-38", "compounds": ["Hard", "Medium"], "prob": 10}
    ]
    if weather["condition"] in ["Rain", "Mixed"]:
        strategies = [
            {"name": "Wet Race (Intermediate -> Intermediate)", "pit_window": "Lap 25-35", "compounds": ["Intermediate", "Intermediate"], "prob": 70},
            {"name": "Heavy Rain (Wet -> Wet)", "pit_window": "Lap 20-30", "compounds": ["Wet", "Wet"], "prob": 30}
        ]
        
    # Generate mock race events
    top_drivers = [r['FullName'] for r in grid_records[:5]]
    mid_drivers = [r['FullName'] for r in grid_records[5:15]]
    
    race_events = []
    race_events.append({"lap": 1, "message": "LIGHTS OUT! And away we go!"})
    
    for i in range(1, 10):
        lap = int(np.random.randint(2, total_laps_race - 2))
        event_type = np.random.choice(["overtake", "pitstop", "fastest_lap", "incident"])
        if event_type == "overtake":
            d1, d2 = np.random.choice(top_drivers + mid_drivers, 2, replace=False)
            race_events.append({"lap": lap, "message": f"Brilliant overtake! {d1} passes {d2}."})
        elif event_type == "pitstop":
            d1 = np.random.choice(top_drivers + mid_drivers)
            race_events.append({"lap": lap, "message": f"{d1} enters the pits for a fresh set of tires."})
        elif event_type == "fastest_lap":
            d1 = np.random.choice(top_drivers)
            race_events.append({"lap": lap, "message": f"New fastest lap by {d1}!"})
        elif event_type == "incident":
            d1 = np.random.choice(mid_drivers)
            race_events.append({"lap": lap, "message": f"Yellow flag! {d1} ran wide but continues."})
            
    race_events = sorted(race_events, key=lambda x: x["lap"])
    race_events.append({"lap": total_laps_race, "message": f"CHEQUERED FLAG! {top_drivers[0]} wins the {track} Grand Prix!"})
    
    return {
        "weather": weather, 
        "grid": grid_records,
        "tire_strategy": strategies,
        "race_events": race_events,
        "total_laps": total_laps_race
    }

@app.post("/api/rag/chat")
def rag_chat(req: ChatQuery):
    query_vec = tfidf_vectorizer.transform([req.query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = int(sims.argmax())
    if sims[best_idx] > 0.1:
        return {"answer": F1_KB[best_idx]}
    return {"answer": "I am a virtual F1 assistant. I specialize in the 2026 F1 regulations and model predictions."}
