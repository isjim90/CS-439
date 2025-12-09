import pandas as pd
import numpy as np
import os
import re

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)



print("\n[STEP 1] Loading datasets from cache")

try:
    with open('dataset_paths.txt', 'r') as f:
        lines = f.readlines()
        dataset1_path = lines[0].split('=')[1].strip()
        dataset2_path = lines[1].split('=')[1].strip()
except FileNotFoundError:
    print("ERROR: dataset_paths.txt not found!")
    exit(1)

players_stats_file = os.path.join(dataset1_path, "players_data_light-2024_2025.csv")
df_stats = pd.read_csv(players_stats_file)
print(f"Loaded player stats: {df_stats.shape[0]} rows x {df_stats.shape[1]} columns")

players_tm_file = os.path.join(dataset2_path, "players.csv")
df_players_tm = pd.read_csv(players_tm_file)
print(f"Loaded Transfermarkt players: {df_players_tm.shape[0]} rows x {df_players_tm.shape[1]} columns")

valuations_file = os.path.join(dataset2_path, "player_valuations.csv")
df_valuations = pd.read_csv(valuations_file)
print(f"Loaded valuations: {df_valuations.shape[0]} rows x {df_valuations.shape[1]} columns")

competitions_file = os.path.join(dataset2_path, "competitions.csv")
df_competitions = pd.read_csv(competitions_file)
print(f"Loaded competitions: {df_competitions.shape[0]} rows x {df_competitions.shape[1]} columns")

print("\n[STEP 2] Cleaning player names with regex")

def clean_player_name(name):
    if pd.isna(name):
        return name
    name = str(name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    name = name.title()
    return name

df_stats['Player_Clean'] = df_stats['Player'].apply(clean_player_name)
print(f"Cleaned {df_stats.shape[0]} player names in stats dataset")

df_players_tm['full_name'] = (df_players_tm['first_name'].fillna('') + ' ' +
                               df_players_tm['last_name'].fillna('')).str.strip()

df_players_tm['full_name_clean'] = df_players_tm['full_name'].apply(clean_player_name)
print(f"Cleaned {df_players_tm.shape[0]} player names in Transfermarkt dataset")

print("\n[STEP 3] Extracting nationality codes with regex")

def extract_nationality(nation_str):
    if pd.isna(nation_str):
        return None
    nation_str = str(nation_str)
    match = re.search(r'[A-Z]{3}', nation_str)
    if match:
        return match.group(0)
    else:
        return None

df_stats['Nationality'] = df_stats['Nation'].apply(extract_nationality)
print(f"Extracted nationality codes for {df_stats['Nationality'].notna().sum()} players")

print("\nSample nationalities extracted:")
print(df_stats[['Player', 'Nation', 'Nationality']].head(10))

print("\n[STEP 4] Cleaning position data...")

def extract_primary_position(pos_str):
    if pd.isna(pos_str):
        return None
    pos_str = str(pos_str)
    primary_pos = pos_str.split(',')[0].strip()
    return primary_pos

df_stats['Position_Primary'] = df_stats['Pos'].apply(extract_primary_position)
print(f"Extracted primary positions")

print("\nPrimary position distribution:")
print(df_stats['Position_Primary'].value_counts())

print("\n[STEP 5] Filtering to top 5 European leagues...")

top_5_leagues = [
    'eng Premier League',
    'es La Liga',
    'it Serie A',
    'de Bundesliga',
    'fr Ligue 1'
]

df_stats_top5 = df_stats[df_stats['Comp'].isin(top_5_leagues)].copy()
print(f"Filtered to top 5 leagues: {df_stats_top5.shape[0]} players (from {df_stats.shape[0]})")

print("\nPlayers per league:")
print(df_stats_top5['Comp'].value_counts())

print("\n[STEP 6] Processing market valuations")

df_valuations['date'] = pd.to_datetime(df_valuations['date'])
df_valuations_sorted = df_valuations.sort_values(['player_id', 'date'], ascending=[True, False])
df_latest_values = df_valuations_sorted.drop_duplicates(subset='player_id', keep='first')
print(f"Got latest market values for {df_latest_values.shape[0]} players")

df_players_with_values = df_players_tm.merge(
    df_latest_values[['player_id', 'market_value_in_eur', 'date']],
    on='player_id',
    how='inner',
    suffixes=('_player', '_valuation')
)
print(f"Merged player info with values: {df_players_with_values.shape[0]} players")

if 'market_value_in_eur_valuation' in df_players_with_values.columns:
    df_players_with_values['market_value_latest'] = df_players_with_values['market_value_in_eur_valuation']
elif 'market_value_in_eur' in df_players_with_values.columns:
    df_players_with_values['market_value_latest'] = df_players_with_values['market_value_in_eur']

print("\n[STEP 7] Merging performance stats with market values...")

merge_cols = ['full_name_clean', 'market_value_latest', 'date',
              'country_of_birth', 'date_of_birth', 'height_in_cm']
merge_cols = [col for col in merge_cols if col in df_players_with_values.columns]

df_merged = df_stats_top5.merge(
    df_players_with_values[merge_cols],
    left_on='Player_Clean',
    right_on='full_name_clean',
    how='inner'
)

df_merged = df_merged.rename(columns={'market_value_latest': 'market_value_in_eur'})

print(f"Merged datasets: {df_merged.shape[0]} players matched (before deduplication)")
print(f"   Match rate: {df_merged.shape[0] / df_stats_top5.shape[0] * 100:.1f}% of top 5 league players")

df_merged = df_merged.drop_duplicates(subset=['Player', 'Squad', 'Comp'], keep='first')
print(f"Removed duplicates: {df_merged.shape[0]} unique players remaining")

print("\n[STEP 8] Handling missing values")

print("Missing values in key columns BEFORE cleaning:")
key_columns = ['Gls', 'Ast', 'Min', 'Age', 'market_value_in_eur', 'xG', 'xAG']
for col in key_columns:
    if col in df_merged.columns:
        missing_count = df_merged[col].isna().sum()
        missing_pct = missing_count / len(df_merged) * 100
        print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")

df_merged = df_merged[df_merged['market_value_in_eur'].notna()].copy()
print(f"\nRemoved rows with missing market values: {df_merged.shape[0]} rows remaining")

df_merged = df_merged[(df_merged['Min'].notna()) & (df_merged['Min'] > 0)].copy()
print(f"Removed rows with no playing time: {df_merged.shape[0]} rows remaining")

numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_merged[col] = df_merged[col].fillna(0)

print(f"Filled remaining missing numeric values with 0")

print("\n[STEP 9] Optimizing memory usage")

memory_before = df_merged.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage BEFORE optimization: {memory_before:.2f} MB")

int_cols = ['Rk', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'CrdY', 'CrdR']
for col in int_cols:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].astype('int32')

categorical_cols = ['Comp', 'Position_Primary', 'Nationality', 'Squad']
for col in categorical_cols:
    if col in df_merged.columns:
        df_merged[col] = df_merged[col].astype('category')

memory_after = df_merged.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage AFTER optimization: {memory_after:.2f} MB")
print(f"Memory saved: {memory_before - memory_after:.2f} MB ({(1 - memory_after/memory_before)*100:.1f}%)")

print("\n[STEP 10] Saving cleaned dataset")

os.makedirs('data', exist_ok=True)

output_file = 'data/cleaned_data.csv'
df_merged.to_csv(output_file, index=False)
print(f"Saved cleaned dataset to: {output_file}")
print(f"   Final dataset: {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")

print("\nFINAL DATASET SUMMARY:")
print(f"   Total players: {df_merged.shape[0]}")
print(f"   Total features: {df_merged.shape[1]}")
print(f"   Leagues: {df_merged['Comp'].nunique()}")
print(f"   Positions: {df_merged['Position_Primary'].nunique()}")
print(f"   Market value range: {df_merged['market_value_in_eur'].min():,.0f} - {df_merged['market_value_in_eur'].max():,.0f}")
print(f"   Average market value: {df_merged['market_value_in_eur'].mean():,.0f}")



