import fastf1
import pandas as pd
import os
import time

# ------------------ CACHE SETUP ------------------
cache_path = r"C:\fastf1_cache"

if not os.path.exists(cache_path):
    os.makedirs(cache_path)

fastf1.Cache.enable_cache(cache_path)

# ------------------ FUNCTION ------------------
def get_qualifying_data(year):
    all_quali_data = []

    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"Schedule loaded for {year}")
    except Exception as e:
        print(f"Could not load schedule for {year}: {e}")
        return pd.DataFrame()

    for _, event in schedule.iterrows():
        try:
            print(f"Loading {event['EventName']} {year}...")

            session = fastf1.get_session(year, event['EventName'], 'Q')
            session.load()

            results = session.results[['DriverNumber', 'FullName', 'TeamName', 'Position']].copy()
            results['Year'] = year
            results['Event'] = event['EventName']

            all_quali_data.append(results)

            time.sleep(2)  # prevents server blocking

        except Exception as e:
            print(f"Skipped {event['EventName']} {year} - {e}")
            continue

    if all_quali_data:
        return pd.concat(all_quali_data, ignore_index=True)
    else:
        return pd.DataFrame()


# ------------------ MAIN EXECUTION ------------------
years = [2022, 2023, 2024, 2025]
all_data = []

for year in years:
    print(f"\nFetching season {year}...")
    season_data = get_qualifying_data(year)

    if not season_data.empty:
        all_data.append(season_data)
    else:
        print(f"No usable data for {year}.")

# Combine everything
if all_data:
    quali_data = pd.concat(all_data, ignore_index=True)
    quali_data.to_csv("qualifying_data.csv", index=False)
    print("\n✅ Data saved successfully as qualifying_data.csv")
else:
    print("❌ No data collected.")
