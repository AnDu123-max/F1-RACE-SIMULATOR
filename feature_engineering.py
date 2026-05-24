import pandas as pd
import numpy as np

# Load collected data
df = pd.read_csv("qualifying_data.csv")

# -----------------------------
# 1. DRIVER AVERAGE QUALIFYING POSITION
# -----------------------------
driver_avg = df.groupby('FullName')['Position'].mean().reset_index()
driver_avg.columns = ['FullName', 'DriverAvgQualiPos']

df = df.merge(driver_avg, on='FullName')

# -----------------------------
# 2. TEAM AVERAGE PERFORMANCE
# -----------------------------
team_avg = df.groupby('TeamName')['Position'].mean().reset_index()
team_avg.columns = ['TeamName', 'TeamAvgQualiPos']

df = df.merge(team_avg, on='TeamName')

# -----------------------------
# 3. DRIVER RECENT FORM (last 5 races)
# -----------------------------
df = df.sort_values(['FullName', 'Year'])

df['RecentForm'] = df.groupby('FullName')['Position'] \
    .rolling(window=5, min_periods=1).mean() \
    .reset_index(0, drop=True)

# -----------------------------
# 4. QUALIFYING SKILL RATING
# Lower avg position = higher rating
# -----------------------------
df['QualifyingSkill'] = 100 - (df['DriverAvgQualiPos'] * 4)

df['QualifyingSkill'] = df['QualifyingSkill'].clip(40, 99)

# -----------------------------
# 5. CONSISTENCY SCORE
# Lower standard deviation = more consistent
# -----------------------------
consistency = df.groupby('FullName')['Position'].std().reset_index()

consistency.columns = ['FullName', 'PositionStd']

df = df.merge(consistency, on='FullName')

df['ConsistencyScore'] = 100 - (df['PositionStd'] * 10)

df['ConsistencyScore'] = df['ConsistencyScore'].fillna(70)
df['ConsistencyScore'] = df['ConsistencyScore'].clip(40, 99)

# -----------------------------
# 6. AGGRESSION SCORE
# More variation in results = aggressive driving
# -----------------------------
df['AggressionScore'] = (
    abs(df['Position'] - df['RecentForm']) * 8
)

df['AggressionScore'] = df['AggressionScore'].clip(20, 99)

# -----------------------------
# 7. TEAM STRENGTH RATING
# -----------------------------
df['TeamStrength'] = 100 - (df['TeamAvgQualiPos'] * 4)

df['TeamStrength'] = df['TeamStrength'].clip(40, 99)

# -----------------------------
# 8. MOMENTUM SCORE
# Better recent form = better momentum
# -----------------------------
df['MomentumScore'] = 100 - (df['RecentForm'] * 4)

df['MomentumScore'] = df['MomentumScore'].clip(40, 99)

# -----------------------------
# 9. DRIVER OVERALL RATING
# Weighted Combination
# -----------------------------
df['DriverRating'] = (
    df['QualifyingSkill'] * 0.35 +
    df['ConsistencyScore'] * 0.25 +
    df['MomentumScore'] * 0.25 +
    df['AggressionScore'] * 0.15
)

df['DriverRating'] = df['DriverRating'].round(1)

# -----------------------------
# 10. TARGET VARIABLE (Pole or Not)
# -----------------------------
df['PolePosition'] = df['Position'].apply(
    lambda x: 1 if x == 1 else 0
)

# -----------------------------
# SAVE DATASET
# -----------------------------
df.to_csv("features_dataset.csv", index=False)

print("Enhanced feature dataset created!")