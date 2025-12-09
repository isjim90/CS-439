import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)



print("\n[STEP 1] Loading data and selecting numeric features")

df = pd.read_csv('data/engineered_features.csv')
print(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features_for_pca = [col for col in numeric_cols
                    if col not in ['market_value_in_eur', 'Log_Market_Value', 'Value_per_Goal']]

X = df[features_for_pca].copy()
X = X.fillna(0)

print(f"Selected {X.shape[1]} numeric features for PCA")
print(f"   Features: {features_for_pca[:10]}")

print("\n[STEP 2] Standardizing features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Standardized {X_scaled.shape[1]} features")
print(f"   Mean of scaled features: {X_scaled.mean():.6f} (should be ~0)")
print(f"   Std of scaled features: {X_scaled.std():.6f} (should be ~1)")

print("\n[STEP 3] Applying PCA")

pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

print(f"Applied PCA: reduced to {X_pca.shape[1]} principal components")

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nVariance Explained by Components:")
for i in range(min(10, len(explained_variance))):
    print(f"   PC{i+1}: {explained_variance[i]*100:.2f}% (Cumulative: {cumulative_variance[i]*100:.2f}%)")

n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nComponents needed for 90% variance: {n_components_90}")

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

print("\n[STEP 4] Creating scree plot")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.bar(range(1, len(explained_variance) + 1), explained_variance * 100,
        color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('Explained Variance by Principal Component', fontsize=14, fontweight='bold')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance (%)', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xlim(0, min(21, len(explained_variance) + 1))

ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100,
         marker='o', linestyle='-', color='coral', linewidth=2, markersize=4)
ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Principal Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% variance')
ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% variance')
ax2.axvline(x=n_components_90, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax2.axvline(x=n_components_95, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax2.legend()
ax2.set_xlim(0, min(31, len(cumulative_variance) + 1))

plt.tight_layout()
plt.savefig('visualizations/07_pca_scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 5] Creating 2D PCA visualization")

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Player': df['Player'],
    'League': df['Comp'],
    'Position': df['Position_Primary'],
    'Market_Value': df['market_value_in_eur']
})

fig, ax = plt.subplots(figsize=(14, 10))
leagues = pca_df['League'].unique()
colors = sns.color_palette("husl", len(leagues))

for league, color in zip(leagues, colors):
    league_data = pca_df[pca_df['League'] == league]
    ax.scatter(league_data['PC1'], league_data['PC2'],
              alpha=0.6, s=50, c=[color], label=league,
              edgecolors='black', linewidth=0.5)

ax.set_title('Players in 2D Principal Component Space (by League)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(title='League', loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('visualizations/08_pca_2d_by_league.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(14, 10))
positions = pca_df['Position'].unique()
colors = sns.color_palette("Set2", len(positions))

for position, color in zip(positions, colors):
    pos_data = pca_df[pca_df['Position'] == position]
    ax.scatter(pos_data['PC1'], pos_data['PC2'],
              alpha=0.6, s=50, c=[color], label=position,
              edgecolors='black', linewidth=0.5)

ax.set_title('Players in 2D Principal Component Space (by Position)',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(title='Position', loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('visualizations/09_pca_2d_by_position.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 6] Analyzing principal component loadings")

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings[:, :5],
    columns=[f'PC{i+1}' for i in range(5)],
    index=features_for_pca
)

print("\nTop 10 Features for Each Principal Component:")
for i in range(min(5, loadings_df.shape[1])):
    pc_name = f'PC{i+1}'
    print(f"\n{pc_name} (explains {explained_variance[i]*100:.2f}% variance):")
    top_features = loadings_df[pc_name].abs().sort_values(ascending=False).head(10)
    for feature, loading in top_features.items():
        actual_loading = loadings_df.loc[feature, pc_name]
        print(f"   {feature}: {actual_loading:.3f}")

loadings_df.to_csv('data/pca_loadings.csv')


print("\n[STEP 7] Creating reduced dataset with top principal components")

n_components_final = 15
pca_final = PCA(n_components=n_components_final)
X_pca_final = pca_final.fit_transform(X_scaled)

pca_features_df = pd.DataFrame(
    X_pca_final,
    columns=[f'PC{i+1}' for i in range(n_components_final)]
)

pca_features_df['market_value_in_eur'] = df['market_value_in_eur'].values
pca_features_df['Log_Market_Value'] = df['Log_Market_Value'].values
pca_features_df['Player'] = df['Player'].values
pca_features_df['League'] = df['Comp'].values
pca_features_df['Position'] = df['Position_Primary'].values

pca_features_df.to_csv('data/pca_features.csv', index=False)
print(f"   Original features: {X.shape[1]}")
print(f"   Reduced to: {n_components_final} principal components")
print(f"   Variance retained: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
print(f"   Dimensionality reduction: {(1 - n_components_final/X.shape[1])*100:.1f}%")

print("\n[STEP 8] Creating component loadings heatmap")

top_features_idx = loadings_df.abs().sum(axis=1).sort_values(ascending=False).head(20).index
loadings_top = loadings_df.loc[top_features_idx, ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

plt.figure(figsize=(10, 12))
sns.heatmap(loadings_top, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Principal Component Loadings (Top 20 Features)',
         fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/10_pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 9] Saving PCA model and scaler")

import pickle

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


with open('data/pca_model.pkl', 'wb') as f:
    pickle.dump(pca_final, f)


print("\n[STEP 10] PCA Summary Statistics")


print(f"   Original dimensions: {X.shape[1]} features")
print(f"   Reduced dimensions: {n_components_final} principal components")
print(f"   Dimensionality reduction: {(1 - n_components_final/X.shape[1])*100:.1f}%")
print(f"   Total variance retained: {pca_final.explained_variance_ratio_.sum()*100:.2f}%")
print(f"   Components for 90% variance: {n_components_90}")
print(f"   Components for 95% variance: {n_components_95}")


print("   1. Scree plot (explained variance)")
print("   2. 2D PCA plot colored by league")
print("   3. 2D PCA plot colored by position")
print("   4. Component loadings heatmap")

