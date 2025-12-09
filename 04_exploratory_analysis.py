import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10



print("\n[STEP 1] Loading engineered features")

df = pd.read_csv('data/engineered_features.csv')
print(f"Loaded engineered features: {df.shape[0]} rows x {df.shape[1]} columns")

os.makedirs('visualizations', exist_ok=True)


print("\n[STEP 2] Calculating correlations with market value")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features_for_correlation = [col for col in numeric_cols
                            if col not in ['market_value_in_eur', 'Log_Market_Value', 'Value_per_Goal']]

correlations = {}
for feature in features_for_correlation:
    corr, p_value = stats.pearsonr(df[feature].fillna(0), df['market_value_in_eur'])
    correlations[feature] = {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

corr_df = pd.DataFrame(correlations).T
corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)

print("\nTop 20 Features Correlated with Market Value:")
print(corr_df.head(20).to_string())

corr_df.to_csv('data/correlation_analysis.csv')
print(f"\nSaved correlation analysis to: data/correlation_analysis.csv")

print("\n[STEP 3] Creating correlation heatmap")

key_features = [
    'market_value_in_eur', 'Age', 'Gls', 'Ast', 'xG', 'xAG',
    'Goals_per_90', 'Assists_per_90', 'Goal_Contributions_per_90',
    'Shot_Accuracy', 'Pass_Completion_Rate', 'Min', 'MP'
]
key_features = [f for f in key_features if f in df.columns]
corr_matrix = df[key_features].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Key Features vs Market Value', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')

plt.close()

print("\n[STEP 4] Creating Age vs Market Value scatter plot")

fig, ax = plt.subplots(figsize=(14, 8))
leagues = df['Comp'].unique()
colors = sns.color_palette("husl", len(leagues))

for league, color in zip(leagues, colors):
    league_data = df[df['Comp'] == league]
    ax.scatter(league_data['Age'],
              league_data['market_value_in_eur'] / 1_000_000,
              alpha=0.6, s=50, c=[color], label=league,
              edgecolors='black', linewidth=0.5)

ax.set_title('Player Age vs Market Value by League', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Market Value (millions)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(title='League', loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('visualizations/02_age_vs_value.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 5] Creating Goals vs Market Value scatter plot")

fig, ax = plt.subplots(figsize=(14, 8))
positions = df['Position_Primary'].unique()
colors = sns.color_palette("Set2", len(positions))

for position, color in zip(positions, colors):
    pos_data = df[df['Position_Primary'] == position]
    ax.scatter(pos_data['Gls'],
              pos_data['market_value_in_eur'] / 1_000_000,
              alpha=0.6, s=50, c=[color], label=position,
              edgecolors='black', linewidth=0.5)

ax.set_title('Goals Scored vs Market Value by Position', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Goals Scored', fontsize=12)
ax.set_ylabel('Market Value (millions)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(title='Position', loc='upper left', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('visualizations/03_goals_vs_value.png', dpi=300, bbox_inches='tight')

plt.close()

print("\n[STEP 6] Creating market value distribution by league")

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df, x='Comp', y='market_value_in_eur', ax=ax, palette='Set3')
ax.set_title('Market Value Distribution by League', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('League', fontsize=12)
ax.set_ylabel('Market Value', fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.xticks(rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('visualizations/04_value_by_league.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 7] Creating top 20 most valuable players chart")

top_20 = df.nlargest(20, 'market_value_in_eur')[['Player', 'market_value_in_eur', 'Comp', 'Position_Primary']]
fig, ax = plt.subplots(figsize=(14, 10))
league_colors = {league: color for league, color in zip(df['Comp'].unique(), sns.color_palette("husl", len(df['Comp'].unique())))}
colors = [league_colors[league] for league in top_20['Comp']]
bars = ax.barh(range(len(top_20)), top_20['market_value_in_eur'] / 1_000_000, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Player'])
ax.set_title('Top 20 Most Valuable Players', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Market Value (millions)', fontsize=12)
ax.set_ylabel('Player', fontsize=12)

for i, (bar, value) in enumerate(zip(bars, top_20['market_value_in_eur'])):
    ax.text(value / 1_000_000 + 1, i, f'{value/1e6:.1f}M', va='center', fontsize=9)

ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/05_top_20_players.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 8] Creating market value distribution histogram")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.hist(df['market_value_in_eur'] / 1_000_000, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('Market Value Distribution (Original)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Market Value (millions)', fontsize=12)
ax1.set_ylabel('Frequency (Number of Players)', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

mean_val = df['market_value_in_eur'].mean() / 1_000_000
median_val = df['market_value_in_eur'].median() / 1_000_000
ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}M')
ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}M')
ax1.legend()

ax2.hist(df['Log_Market_Value'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_title('Market Value Distribution (Log-Transformed)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Log(Market Value)', fontsize=12)
ax2.set_ylabel('Frequency (Number of Players)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

mean_log = df['Log_Market_Value'].mean()
median_log = df['Log_Market_Value'].median()
ax2.axvline(mean_log, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_log:.2f}')
ax2.axvline(median_log, color='green', linestyle='--', linewidth=2, label=f'Median: {median_log:.2f}')
ax2.legend()

plt.tight_layout()
plt.savefig('visualizations/06_value_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 9] Calculating summary statistics by league")

league_stats = df.groupby('Comp').agg({
    'market_value_in_eur': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'Gls': 'mean',
    'Ast': 'mean',
    'Age': 'mean'
}).round(2)

league_stats.columns = ['_'.join(col).strip() for col in league_stats.columns.values]
league_stats = league_stats.rename(columns={
    'market_value_in_eur_count': 'Num_Players',
    'market_value_in_eur_mean': 'Avg_Value',
    'market_value_in_eur_median': 'Median_Value',
    'market_value_in_eur_std': 'Std_Value',
    'market_value_in_eur_min': 'Min_Value',
    'market_value_in_eur_max': 'Max_Value',
    'Gls_mean': 'Avg_Goals',
    'Ast_mean': 'Avg_Assists',
    'Age_mean': 'Avg_Age'
})

print("\nSummary Statistics by League:")
print(league_stats.to_string())

league_stats.to_csv('data/league_statistics.csv')


print("\n[STEP 10] Calculating summary statistics by position")

position_stats = df.groupby('Position_Primary').agg({
    'market_value_in_eur': ['count', 'mean', 'median'],
    'Gls': 'mean',
    'Ast': 'mean',
    'Goals_per_90': 'mean',
    'Assists_per_90': 'mean'
}).round(2)

position_stats.columns = ['_'.join(col).strip() for col in position_stats.columns.values]

print("\nSummary Statistics by Position:")
print(position_stats.to_string())

position_stats.to_csv('data/position_statistics.csv')



print("   1. Correlation heatmap")
print("   2. Age vs Market Value scatter plot")
print("   3. Goals vs Market Value scatter plot")
print("   4. Market Value by League box plot")
print("   5. Top 20 Most Valuable Players bar chart")
print("   6. Market Value distribution histograms")

