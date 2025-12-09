import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)



print("\n[STEP 1] Loading predictions and data")

predictions = pd.read_csv('data/predictions.csv')
print(f"Loaded predictions: {predictions.shape[0]} players")

df = pd.read_csv('data/engineered_features.csv')
print(f"Loaded features: {df.shape[0]} players")

predictions_full = predictions.merge(
    df[['Player', 'Comp', 'Position_Primary', 'Age', 'Gls', 'Ast', 'Squad']],
    on='Player',
    how='left'
)

print(f"Merged data: {predictions_full.shape[0]} players with full information")

print("\n[STEP 2] Identifying undervalued players")

predictions_full['Value_Difference'] = predictions_full['Predicted_Value'] - predictions_full['Actual_Value']
predictions_full['Value_Ratio'] = predictions_full['Predicted_Value'] / predictions_full['Actual_Value']

undervalued_all = predictions_full[predictions_full['Value_Difference'] > 0].copy()
undervalued_all = undervalued_all.sort_values('Value_Difference', ascending=False)

undervalued_top20 = undervalued_all.head(20)

print(f"\nTOP 20 MOST UNDERVALUED PLAYERS (out of {len(undervalued_all)} total):")


undervalued_display = undervalued_top20[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age', 'Gls', 'Ast',
    'Actual_Value', 'Predicted_Value', 'Value_Difference'
]].copy()

undervalued_display['Actual_Value_M'] = undervalued_display['Actual_Value'] / 1_000_000
undervalued_display['Predicted_Value_M'] = undervalued_display['Predicted_Value'] / 1_000_000
undervalued_display['Difference_M'] = undervalued_display['Value_Difference'] / 1_000_000

print(undervalued_display[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age',
    'Actual_Value_M', 'Predicted_Value_M', 'Difference_M'
]].to_string(index=False))

undervalued_all_display = undervalued_all[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age', 'Gls', 'Ast',
    'Actual_Value', 'Predicted_Value', 'Value_Difference'
]].copy()
undervalued_all_display['Actual_Value_M'] = undervalued_all_display['Actual_Value'] / 1_000_000
undervalued_all_display['Predicted_Value_M'] = undervalued_all_display['Predicted_Value'] / 1_000_000
undervalued_all_display['Difference_M'] = undervalued_all_display['Value_Difference'] / 1_000_000

undervalued_all_display.to_csv('data/undervalued_players.csv', index=False)
print(f"\nSaved ALL {len(undervalued_all)} undervalued players to: data/undervalued_players.csv")

print("\n[STEP 3] Identifying overvalued players")

overvalued_all = predictions_full[predictions_full['Value_Difference'] < 0].copy()
overvalued_all = overvalued_all.sort_values('Value_Difference', ascending=True)

overvalued_top20 = overvalued_all.head(20)

print(f"\nTOP 20 MOST OVERVALUED PLAYERS (out of {len(overvalued_all)} total):")

overvalued_display = overvalued_top20[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age', 'Gls', 'Ast',
    'Actual_Value', 'Predicted_Value', 'Value_Difference'
]].copy()

overvalued_display['Actual_Value_M'] = overvalued_display['Actual_Value'] / 1_000_000
overvalued_display['Predicted_Value_M'] = overvalued_display['Predicted_Value'] / 1_000_000
overvalued_display['Difference_M'] = overvalued_display['Value_Difference'] / 1_000_000

print(overvalued_display[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age',
    'Actual_Value_M', 'Predicted_Value_M', 'Difference_M'
]].to_string(index=False))

overvalued_all_display = overvalued_all[[
    'Player', 'Squad', 'Comp', 'Position_Primary', 'Age', 'Gls', 'Ast',
    'Actual_Value', 'Predicted_Value', 'Value_Difference'
]].copy()
overvalued_all_display['Actual_Value_M'] = overvalued_all_display['Actual_Value'] / 1_000_000
overvalued_all_display['Predicted_Value_M'] = overvalued_all_display['Predicted_Value'] / 1_000_000
overvalued_all_display['Difference_M'] = overvalued_all_display['Value_Difference'] / 1_000_000

overvalued_all_display.to_csv('data/overvalued_players.csv', index=False)
print(f"\nSaved ALL {len(overvalued_all)} overvalued players to: data/overvalued_players.csv")

print("\n[STEP 4] Analyzing valuation patterns by league")

league_analysis = predictions_full.groupby('Comp').agg({
    'Value_Difference': ['mean', 'median', 'std'],
    'Value_Ratio': ['mean', 'median'],
    'Player': 'count'
}).round(2)

league_analysis.columns = ['_'.join(col).strip() for col in league_analysis.columns.values]
league_analysis = league_analysis.rename(columns={
    'Value_Difference_mean': 'Avg_Difference',
    'Value_Difference_median': 'Median_Difference',
    'Value_Difference_std': 'Std_Difference',
    'Value_Ratio_mean': 'Avg_Ratio',
    'Value_Ratio_median': 'Median_Ratio',
    'Player_count': 'Num_Players'
})

league_analysis = league_analysis.sort_values('Avg_Difference', ascending=False)

print("\nLEAGUE VALUATION ANALYSIS:")
print("(Positive difference = league undervalues players on average)")
print("(Negative difference = league overvalues players on average)")
print(league_analysis.to_string())

league_analysis.to_csv('data/league_valuation_analysis.csv')

fig, ax = plt.subplots(figsize=(12, 6))
leagues = league_analysis.index
avg_diff = league_analysis['Avg_Difference'] / 1_000_000

colors = ['green' if x > 0 else 'red' for x in avg_diff]

ax.bar(leagues, avg_diff, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_title('Average Valuation Difference by League', fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Difference: Predicted - Actual (millions)', fontsize=12)
ax.set_xlabel('League', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('visualizations/14_league_valuation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 5] Analyzing valuation patterns by position")

position_analysis = predictions_full.groupby('Position_Primary').agg({
    'Value_Difference': ['mean', 'median'],
    'Value_Ratio': ['mean', 'median'],
    'Player': 'count',
    'Actual_Value': 'mean',
    'Predicted_Value': 'mean'
}).round(2)

position_analysis.columns = ['_'.join(col).strip() for col in position_analysis.columns.values]
position_analysis = position_analysis.rename(columns={
    'Value_Difference_mean': 'Avg_Difference',
    'Value_Difference_median': 'Median_Difference',
    'Value_Ratio_mean': 'Avg_Ratio',
    'Value_Ratio_median': 'Median_Ratio',
    'Player_count': 'Num_Players',
    'Actual_Value_mean': 'Avg_Actual_Value',
    'Predicted_Value_mean': 'Avg_Predicted_Value'
})

print("\nPOSITION VALUATION ANALYSIS:")
print(position_analysis.to_string())

position_analysis.to_csv('data/position_valuation_analysis.csv')

print("\n[STEP 6] Analyzing valuation patterns by age")

predictions_full['Age_Bin'] = pd.cut(
    predictions_full['Age'],
    bins=[0, 23, 26, 29, 100],
    labels=['Under 23', '23-26', '27-29', '30+']
)

age_analysis = predictions_full.groupby('Age_Bin').agg({
    'Value_Difference': ['mean', 'median'],
    'Actual_Value': 'mean',
    'Predicted_Value': 'mean',
    'Player': 'count'
}).round(2)

age_analysis.columns = ['_'.join(col).strip() for col in age_analysis.columns.values]

print("\nAGE GROUP VALUATION ANALYSIS:")
print(age_analysis.to_string())

age_analysis.to_csv('data/age_valuation_analysis.csv')


print("\n[STEP 7] Finding best value young players")

young_players = predictions_full[predictions_full['Age'] < 23].copy()
young_undervalued = young_players.nlargest(15, 'Value_Ratio')

print("\nTOP 15 UNDERVALUED YOUNG PLAYERS (Under 23):")


young_display = young_undervalued[[
    'Player', 'Squad', 'Comp', 'Age', 'Gls', 'Ast',
    'Actual_Value', 'Predicted_Value', 'Value_Ratio'
]].copy()

young_display['Actual_Value_M'] = young_display['Actual_Value'] / 1_000_000
young_display['Predicted_Value_M'] = young_display['Predicted_Value'] / 1_000_000

print(young_display[[
    'Player', 'Squad', 'Age', 'Actual_Value_M', 'Predicted_Value_M', 'Value_Ratio'
]].to_string(index=False))

young_display.to_csv('data/young_undervalued_players.csv', index=False)

print("\n[STEP 8] Creating comprehensive summary report")

total_players = len(predictions_full)
avg_actual_value = predictions_full['Actual_Value'].mean()
avg_predicted_value = predictions_full['Predicted_Value'].mean()
avg_error = predictions_full['Abs_Error'].mean()
median_error = predictions_full['Abs_Error'].median()

num_undervalued = (predictions_full['Value_Difference'] > 0).sum()
num_overvalued = (predictions_full['Value_Difference'] < 0).sum()

model_comparison = pd.read_csv('data/model_comparison.csv')
best_model = model_comparison.iloc[0]

r2_col = 'R2' if 'R2' in model_comparison.columns else 'R²'

summary_report = f"""
Predicting Player Market Value in Europe's Top 5 Football Leagues


1. DATASET SUMMARY

Total Players Analyzed: {total_players}
Average Actual Market Value: {avg_actual_value:,.0f}
Average Predicted Market Value: {avg_predicted_value:,.0f}
Average Prediction Error: {avg_error:,.0f}
Median Prediction Error: {median_error:,.0f}


2. MODEL PERFORMANCE


Best Model: {best_model['Model']}
R2 Score: {best_model[r2_col]:.4f}
Mean Squared Error: {best_model['MSE']:,.0f}
Mean Absolute Error: {best_model['MAE']:,.0f}

All Models Tested:
{model_comparison.to_string(index=False)}


3. KEY FINDINGS


Undervalued Players: {num_undervalued} ({num_undervalued/total_players*100:.1f}%)
Overvalued Players: {num_overvalued} ({num_overvalued/total_players*100:.1f}%)

Top 5 Most Undervalued Players:
{undervalued_display[['Player', 'Squad', 'Actual_Value_M', 'Predicted_Value_M']].head().to_string(index=False)}


4. LEAGUE INSIGHTS

{league_analysis[['Avg_Difference', 'Avg_Ratio', 'Num_Players']].to_string()}


5. POSITION INSIGHTS

{position_analysis[['Avg_Difference', 'Avg_Actual_Value', 'Num_Players']].to_string()}




7. COURSE CONCEPTS APPLIED


Regex: Cleaned player names, extracted nationality codes
Linear Regression: Univariate and multivariate models
Gradient Descent: Manual implementation with convergence
Feature Engineering: Created 15+ derived features
Correlation Analysis: Identified top predictive features
PCA/SVD: Reduced 50+ features to 15 components
Data Visualization: Created 14 plots and charts
Pandas Optimization: Memory reduction, efficient data types
Model Complexity: Train-test split, cross-validation, regularization
Evaluation Metrics: R2, MSE, MAE, cross-validation scores


END OF REPORT

"""

with open('data/FINAL_SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)


print("\n[STEP 9] Creating final visualization")

fig, ax = plt.subplots(figsize=(14, 10))

under = predictions_full[predictions_full['Value_Difference'] > 0]
over = predictions_full[predictions_full['Value_Difference'] < 0]

ax.scatter(under['Actual_Value'] / 1_000_000,
          under['Predicted_Value'] / 1_000_000,
          alpha=0.6, s=50, c='green', label='Undervalued', edgecolors='black', linewidth=0.5)

ax.scatter(over['Actual_Value'] / 1_000_000,
          over['Predicted_Value'] / 1_000_000,
          alpha=0.6, s=50, c='red', label='Overvalued', edgecolors='black', linewidth=0.5)

max_val = max(predictions_full['Actual_Value'].max(), predictions_full['Predicted_Value'].max()) / 1_000_000
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Prediction')

ax.set_title('Player Valuation Analysis: Undervalued vs Overvalued', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Actual Market Value (millions)', fontsize=12)
ax.set_ylabel('Predicted Market Value (millions)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/15_undervalued_vs_overvalued.png', dpi=300, bbox_inches='tight')
plt.close()


print("   - data/undervalued_players.csv")
print("   - data/overvalued_players.csv")
print("   - data/young_undervalued_players.csv")
print("   - data/league_valuation_analysis.csv")
print("   - data/position_valuation_analysis.csv")
print("   - data/FINAL_SUMMARY_REPORT.txt")
print("   - visualizations/14_league_valuation_analysis.png")
print("   - visualizations/15_undervalued_vs_overvalued.png")

