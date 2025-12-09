import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)



print("\n[STEP 1] Loading Dataset 1: Player Performance Stats (2024-2025)")

try:
    with open('dataset_paths.txt', 'r') as f:
        lines = f.readlines()
        dataset1_path = lines[0].split('=')[1].strip()
        dataset2_path = lines[1].split('=')[1].strip()
except FileNotFoundError:
    print("ERROR: dataset_paths.txt not found!")
    exit(1)

players_full_path = os.path.join(dataset1_path, "players_data-2024_2025.csv")
df_players_full = pd.read_csv(players_full_path)

players_light_path = os.path.join(dataset1_path, "players_data_light-2024_2025.csv")
df_players_light = pd.read_csv(players_light_path)

print(f"Loaded full dataset: {df_players_full.shape[0]} rows x {df_players_full.shape[1]} columns")
print(f"Loaded light dataset: {df_players_light.shape[0]} rows x {df_players_light.shape[1]} columns")

print("\nFirst 5 rows of player data:")
print(df_players_light.head())

print("\nAvailable columns in light dataset:")
print(df_players_light.columns.tolist())

print("\nMissing values in key columns:")
missing_counts = df_players_light.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("No missing values found!")

print("\nLeagues in dataset:")
if 'Comp' in df_players_light.columns:
    print(df_players_light['Comp'].value_counts())

print("\nPositions in dataset:")
if 'Pos' in df_players_light.columns:
    print(df_players_light['Pos'].value_counts())

print("\n[STEP 2] Loading Dataset 2: Transfermarkt Market Values")

players_tm_path = os.path.join(dataset2_path, "players.csv")
df_players_tm = pd.read_csv(players_tm_path)
print(f"Loaded players table: {df_players_tm.shape[0]} rows x {df_players_tm.shape[1]} columns")

valuations_path = os.path.join(dataset2_path, "player_valuations.csv")
df_valuations = pd.read_csv(valuations_path)
print(f"Loaded valuations table: {df_valuations.shape[0]} rows x {df_valuations.shape[1]} columns")

competitions_path = os.path.join(dataset2_path, "competitions.csv")
df_competitions = pd.read_csv(competitions_path)
print(f"Loaded competitions table: {df_competitions.shape[0]} rows x {df_competitions.shape[1]} columns")

clubs_path = os.path.join(dataset2_path, "clubs.csv")
df_clubs = pd.read_csv(clubs_path)
print(f"Loaded clubs table: {df_clubs.shape[0]} rows x {df_clubs.shape[1]} columns")

print("\nFirst 5 rows of player valuations:")
print(df_valuations.head())

print("\nColumns in valuations table:")
print(df_valuations.columns.tolist())

print("\nColumns in players table:")
print(df_players_tm.columns.tolist())

print("\nAvailable competitions:")
print(df_competitions[['competition_id', 'name', 'country_name']].head(20))

print("\n[STEP 3] Summary Statistics")

print("\nSummary statistics for goals and assists:")
if 'Gls' in df_players_light.columns and 'Ast' in df_players_light.columns:
    print(df_players_light[['Gls', 'Ast', 'Min']].describe())

print("\nSummary statistics for market values:")
if 'market_value_in_eur' in df_valuations.columns:
    print(df_valuations['market_value_in_eur'].describe())



