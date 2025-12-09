import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)



print("\n[STEP 1] Loading cleaned dataset")

df = pd.read_csv('data/cleaned_data.csv')
print(f"Loaded cleaned data: {df.shape[0]} rows x {df.shape[1]} columns")

print("\nFirst 5 rows:")
print(df[['Player', 'Comp', 'Position_Primary', 'Gls', 'Ast', 'Min', 'market_value_in_eur']].head())

print("\n[STEP 2] Creating efficiency metrics")

df['90s'] = df['Min'] / 90.0
print(f"Created '90s' feature (full matches played)")

df['Goals_per_90'] = df['Gls'] / (df['90s'] + 0.01)
print(f"Created 'Goals_per_90' feature")

df['Assists_per_90'] = df['Ast'] / (df['90s'] + 0.01)
print(f"Created 'Assists_per_90' feature")

df['Goal_Contributions_per_90'] = df['Goals_per_90'] + df['Assists_per_90']
print(f"Created 'Goal_Contributions_per_90' feature")

if 'xG' in df.columns:
    df['xG_per_90'] = df['xG'] / (df['90s'] + 0.01)
    print(f"Created 'xG_per_90' feature")

if 'xAG' in df.columns:
    df['xAG_per_90'] = df['xAG'] / (df['90s'] + 0.01)
    print(f"Created 'xAG_per_90' feature")

if 'Sh' in df.columns:
    df['Shots_per_90'] = df['Sh'] / (df['90s'] + 0.01)
    print(f"Created 'Shots_per_90' feature")

if 'SoT' in df.columns and 'Sh' in df.columns:
    df['Shot_Accuracy'] = df['SoT'] / (df['Sh'] + 0.01)
    df['Shot_Accuracy'] = df['Shot_Accuracy'].clip(upper=1.0)
    print(f"Created 'Shot_Accuracy' feature")

if 'Cmp' in df.columns and 'Att' in df.columns:
    df['Pass_Completion_Rate'] = df['Cmp'] / (df['Att'] + 0.01)
    df['Pass_Completion_Rate'] = df['Pass_Completion_Rate'].clip(upper=1.0)
    print(f"Created 'Pass_Completion_Rate' feature")

if 'Tkl' in df.columns:
    df['Tackles_per_90'] = df['Tkl'] / (df['90s'] + 0.01)
    print(f"Created 'Tackles_per_90' feature")

if 'Int' in df.columns:
    df['Interceptions_per_90'] = df['Int'] / (df['90s'] + 0.01)
    print(f"Created 'Interceptions_per_90' feature")

print("\n[STEP 3] Creating age-based features")

def categorize_age(age):
    if pd.isna(age):
        return 'Unknown'
    elif age < 23:
        return 'Young'
    elif age <= 29:
        return 'Prime'
    else:
        return 'Veteran'

df['Age_Group'] = df['Age'].apply(categorize_age)
print(f"Created 'Age_Group' feature")

print("\nAge group distribution:")
print(df['Age_Group'].value_counts())

df['Age_Squared'] = df['Age'] ** 2
print(f"Created 'Age_Squared' feature for polynomial regression")

print("\n[STEP 4] Creating league dummy variables")

league_dummies = pd.get_dummies(df['Comp'], prefix='League')
print(f"Created {league_dummies.shape[1]} league dummy variables:")
print(f"   {list(league_dummies.columns)}")

df = pd.concat([df, league_dummies], axis=1)

print("\n[STEP 5] Creating position dummy variables")

position_dummies = pd.get_dummies(df['Position_Primary'], prefix='Position')
print(f"Created {position_dummies.shape[1]} position dummy variables:")
print(f"   {list(position_dummies.columns)}")

df = pd.concat([df, position_dummies], axis=1)

print("\n[STEP 6] Creating interaction features")

df['Age_x_Goals'] = df['Age'] * df['Gls']
print(f"Created 'Age_x_Goals' interaction feature")

df['Minutes_x_Goals_per_90'] = df['Min'] * df['Goals_per_90']
print(f"Created 'Minutes_x_Goals_per_90' interaction feature")

print("\n[STEP 7] Creating value-based features")

df['Log_Market_Value'] = np.log1p(df['market_value_in_eur'])
print(f"Created 'Log_Market_Value' feature (log-transformed target)")

df['Value_per_Goal'] = df['market_value_in_eur'] / (df['Gls'] + 1)
print(f"Created 'Value_per_Goal' feature")

print("\n[STEP 8] Handling infinite and extreme values")

df = df.replace([np.inf, -np.inf], np.nan)
inf_count = df.isna().sum().sum() - df.isna().sum().sum()


engineered_features = [
    'Goals_per_90', 'Assists_per_90', 'Goal_Contributions_per_90',
    'xG_per_90', 'xAG_per_90', 'Shots_per_90', 'Shot_Accuracy',
    'Pass_Completion_Rate', 'Tackles_per_90', 'Interceptions_per_90',
    'Value_per_Goal'
]

for feature in engineered_features:
    if feature in df.columns:
        df[feature] = df[feature].fillna(0)



for feature in engineered_features:
    if feature in df.columns:
        p99 = df[feature].quantile(0.99)
        df[feature] = df[feature].clip(upper=p99)



print("\n[STEP 9] Selecting final feature set")

feature_columns = [
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age_Group',
    'Age', 'Min', 'MP', 'Starts', 'Gls', 'Ast', 'xG', 'xAG',
    '90s', 'Goals_per_90', 'Assists_per_90', 'Goal_Contributions_per_90',
    'xG_per_90', 'xAG_per_90', 'Shots_per_90', 'Shot_Accuracy',
    'Pass_Completion_Rate', 'Tackles_per_90', 'Interceptions_per_90',
    'Age_Squared',
    'Age_x_Goals', 'Minutes_x_Goals_per_90',
    'market_value_in_eur', 'Log_Market_Value', 'Value_per_Goal'
]

feature_columns.extend(league_dummies.columns.tolist())
feature_columns.extend(position_dummies.columns.tolist())
feature_columns = [col for col in feature_columns if col in df.columns]
df_features = df[feature_columns].copy()

print(f"Selected {len(feature_columns)} features for modeling")

print("\n[STEP 10] Saving engineered features")

output_file = 'data/engineered_features.csv'
df_features.to_csv(output_file, index=False)
print(f"Saved engineered features to: {output_file}")
print(f"   Final dataset: {df_features.shape[0]} rows x {df_features.shape[1]} columns")

print("\nENGINEERED FEATURES SUMMARY:")
print(f"   Total features: {df_features.shape[1]}")
print(f"   Efficiency metrics: 11")
print(f"   League dummies: {len(league_dummies.columns)}")
print(f"   Position dummies: {len(position_dummies.columns)}")
print(f"   Age features: 2")
print(f"   Interaction features: 2")

print("\nKey Feature Statistics:")
key_features = ['Goals_per_90', 'Assists_per_90', 'Goal_Contributions_per_90',
                'Shot_Accuracy', 'Pass_Completion_Rate']
for feature in key_features:
    if feature in df_features.columns:
        print(f"   {feature}:")
        print(f"      Mean: {df_features[feature].mean():.3f}")
        print(f"      Std:  {df_features[feature].std():.3f}")
        print(f"      Min:  {df_features[feature].min():.3f}")
        print(f"      Max:  {df_features[feature].max():.3f}")



