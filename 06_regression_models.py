import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)



print("\n[STEP 1] Loading data and preparing features")

df = pd.read_csv('data/engineered_features.csv')
print(f"Loaded data: {df.shape[0]} rows x {df.shape[1]} columns")

feature_cols = [
    'Age', 'Age_Squared', 'Min', 'MP', 'Gls', 'Ast', 'xG', 'xAG',
    'Goals_per_90', 'Assists_per_90', 'Goal_Contributions_per_90',
    'xG_per_90', 'xAG_per_90', 'Shots_per_90', 'Shot_Accuracy',
    'Pass_Completion_Rate', 'Tackles_per_90', 'Interceptions_per_90'
]

league_cols = [col for col in df.columns if col.startswith('League_')]
feature_cols.extend(league_cols)

position_cols = [col for col in df.columns if col.startswith('Position_')]
feature_cols.extend(position_cols)

feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df['market_value_in_eur'].copy()
X = X.fillna(0)
X = X.select_dtypes(include=[np.number])

print(f"Selected {X.shape[1]} features for modeling")
print(f"   Target variable: market_value_in_eur")
print(f"   Target range: {y.min():,.0f} to {y.max():,.0f}")
print(f"   Target mean: {y.mean():,.0f}")

print("\n[STEP 2] Splitting data into train and test sets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Split data:")
print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\n[STEP 3] Standardizing features")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Standardized features")
print(f"   Training mean: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"   Training std: {X_train_scaled.std():.6f} (should be ~1)")

print("\n[STEP 4] Building univariate linear regression models")

univariate_results = []
key_features = ['Age', 'Gls', 'Ast', 'Goals_per_90', 'Assists_per_90',
                'Goal_Contributions_per_90', 'xG', 'xAG', 'Min']

for feature in key_features:
    if feature in X_train.columns:
        feature_idx = X_train.columns.get_loc(feature)
        X_train_single = X_train_scaled[:, feature_idx].reshape(-1, 1)
        X_test_single = X_test_scaled[:, feature_idx].reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_train_single, y_train)
        y_pred = model.predict(X_test_single)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        univariate_results.append({
            'Feature': feature,
            'R2': r2,
            'MSE': mse,
            'MAE': mae,
            'Coefficient': model.coef_[0],
            'Intercept': model.intercept_
        })

univariate_df = pd.DataFrame(univariate_results)
univariate_df = univariate_df.sort_values('R2', ascending=False)

print("\nUnivariate Regression Results (sorted by R2):")
print(univariate_df.to_string(index=False))

univariate_df.to_csv('data/univariate_regression_results.csv', index=False)


print("\n[STEP 5] Building multivariate linear regression model")

model_multi = LinearRegression()
model_multi.fit(X_train_scaled, y_train)

y_train_pred = model_multi.predict(X_train_scaled)
y_test_pred = model_multi.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nMultivariate Linear Regression Results:")
print(f"   Training Set:")
print(f"      R2 Score: {train_r2:.4f}")
print(f"      MSE: {train_mse:,.0f}")
print(f"      MAE: {train_mae:,.0f}")
print(f"\n   Test Set:")
print(f"      R2 Score: {test_r2:.4f}")
print(f"      MSE: {test_mse:,.0f}")
print(f"      MAE: {test_mae:,.0f}")

overfitting_gap = train_r2 - test_r2
print(f"\n   Overfitting Check:")
print(f"      R2 Gap (train - test): {overfitting_gap:.4f}")
if overfitting_gap > 0.1:
    print(f"      Warning: Possible overfitting (gap > 0.1)")
else:
    print(f"      Good generalization (gap < 0.1)")

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model_multi.coef_
})
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features (by coefficient magnitude):")
print(feature_importance.head(10)[['Feature', 'Coefficient']].to_string(index=False))

feature_importance.to_csv('data/feature_importance.csv', index=False)

print("\n[STEP 6] Implementing gradient descent manually")

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, verbose=True):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    cost_history = []

    for iteration in range(n_iterations):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        cost = (1 / n_samples) * np.sum(error ** 2)
        cost_history.append(cost)
        dw = (2 / n_samples) * np.dot(X.T, error)
        db = (2 / n_samples) * np.sum(error)
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        if verbose and (iteration % 100 == 0 or iteration == n_iterations - 1):
            print(f"   Iteration {iteration}: MSE = {cost:,.0f}")

    return weights, bias, cost_history

print("\nRunning gradient descent (1000 iterations)")
weights_gd, bias_gd, cost_history = gradient_descent(
    X_train_scaled,
    y_train.values,
    learning_rate=0.01,
    n_iterations=1000,
    verbose=True
)

y_train_pred_gd = np.dot(X_train_scaled, weights_gd) + bias_gd
y_test_pred_gd = np.dot(X_test_scaled, weights_gd) + bias_gd

gd_train_r2 = r2_score(y_train, y_train_pred_gd)
gd_test_r2 = r2_score(y_test, y_test_pred_gd)
gd_test_mse = mean_squared_error(y_test, y_test_pred_gd)
gd_test_mae = mean_absolute_error(y_test, y_test_pred_gd)

print("\nGradient Descent Model Results:")
print(f"   Training R2: {gd_train_r2:.4f}")
print(f"   Test R2: {gd_test_r2:.4f}")
print(f"   Test MSE: {gd_test_mse:,.0f}")
print(f"   Test MAE: {gd_test_mae:,.0f}")

print("\nComparison: Gradient Descent vs Sklearn:")
print(f"   R2 difference: {abs(gd_test_r2 - test_r2):.6f} ")
print(f"   Models match!" if abs(gd_test_r2 - test_r2) < 0.01 else "   Models differ")

plt.figure(figsize=(12, 6))
plt.plot(cost_history, linewidth=2, color='blue')
plt.title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (MSE)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/11_gradient_descent_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n[STEP 7] Building polynomial regression model")

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = ['Age', 'Gls', 'Ast', 'Min']
poly_feature_idx = [X_train.columns.get_loc(f) for f in poly_features if f in X_train.columns]

X_train_poly_subset = X_train_scaled[:, poly_feature_idx]
X_test_poly_subset = X_test_scaled[:, poly_feature_idx]

X_train_poly = poly.fit_transform(X_train_poly_subset)
X_test_poly = poly.transform(X_test_poly_subset)

print(f"Created polynomial features:")
print(f"   Original features: {X_train_poly_subset.shape[1]}")
print(f"   Polynomial features: {X_train_poly.shape[1]}")

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

y_test_pred_poly = model_poly.predict(X_test_poly)

poly_r2 = r2_score(y_test, y_test_pred_poly)
poly_mse = mean_squared_error(y_test, y_test_pred_poly)
poly_mae = mean_absolute_error(y_test, y_test_pred_poly)

print("\nPolynomial Regression Results:")
print(f"   R2 Score: {poly_r2:.4f}")
print(f"   MSE: {poly_mse:,.0f}")
print(f"   MAE: {poly_mae:,.0f}")

print("\n[STEP 8] Building regularized regression models")

model_ridge = Ridge(alpha=1000.0)
model_ridge.fit(X_train_scaled, y_train)
y_test_pred_ridge = model_ridge.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, y_test_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_test_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_test_pred_ridge)

print("\nRidge Regression Results (L2 Regularization):")
print(f"   R2 Score: {ridge_r2:.4f}")
print(f"   MSE: {ridge_mse:,.0f}")
print(f"   MAE: {ridge_mae:,.0f}")

model_lasso = Lasso(alpha=100.0)
model_lasso.fit(X_train_scaled, y_train)
y_test_pred_lasso = model_lasso.predict(X_test_scaled)

lasso_r2 = r2_score(y_test, y_test_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_test_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_test_pred_lasso)

print("\nLasso Regression Results (L1 Regularization):")
print(f"   R2 Score: {lasso_r2:.4f}")
print(f"   MSE: {lasso_mse:,.0f}")
print(f"   MAE: {lasso_mae:,.0f}")

non_zero_coefs = np.sum(model_lasso.coef_ != 0)
print(f"   Features selected: {non_zero_coefs} out of {len(model_lasso.coef_)}")

print("\n[STEP 9] Performing cross-validation")

cv_scores = cross_val_score(
    model_multi,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='r2'
)

print("\n5-Fold Cross-Validation Results:")
print(f"   Fold scores: {cv_scores}")
print(f"   Mean R2: {cv_scores.mean():.4f}")
print(f"   Std R2: {cv_scores.std():.4f}")
print(f"   95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

print("\n[STEP 10] Comparing all models")

model_comparison = pd.DataFrame({
    'Model': ['Multivariate Linear', 'Gradient Descent', 'Polynomial', 'Ridge', 'Lasso'],
    'R2': [test_r2, gd_test_r2, poly_r2, ridge_r2, lasso_r2],
    'MSE': [test_mse, gd_test_mse, poly_mse, ridge_mse, lasso_mse],
    'MAE': [test_mae, gd_test_mae, poly_mae, ridge_mae, lasso_mae]
})

model_comparison = model_comparison.sort_values('R2', ascending=False)

print("\nMODEL COMPARISON (sorted by R2):")
print(model_comparison.to_string(index=False))

model_comparison.to_csv('data/model_comparison.csv', index=False)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].bar(model_comparison['Model'], model_comparison['R2'], color='skyblue', edgecolor='black')
axes[0].set_title('R2 Score Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R2 Score', fontsize=12)
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(model_comparison['Model'], model_comparison['MSE'], color='lightcoral', edgecolor='black')
axes[1].set_title('MSE Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('MSE', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(model_comparison['Model'], model_comparison['MAE'], color='lightgreen', edgecolor='black')
axes[2].set_title('MAE Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylabel('MAE', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/12_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(y_test / 1_000_000, y_test_pred / 1_000_000,
          alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

max_val = max(y_test.max(), y_test_pred.max()) / 1_000_000
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_title(f'Predicted vs Actual Market Value\n(Multivariate Linear Regression, R2={test_r2:.3f})',
            fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Market Value (millions)', fontsize=12)
ax.set_ylabel('Predicted Market Value (millions)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/13_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

predictions_df = pd.DataFrame({
    'Player': df.loc[X_test.index, 'Player'].values,
    'Actual_Value': y_test.values,
    'Predicted_Value': y_test_pred,
    'Error': y_test.values - y_test_pred,
    'Abs_Error': np.abs(y_test.values - y_test_pred),
    'Percent_Error': np.abs((y_test.values - y_test_pred) / y_test.values) * 100
})

predictions_df.to_csv('data/predictions.csv', index=False)

print(f"   Best Model: {model_comparison.iloc[0]['Model']}")
print(f"   Best R2: {model_comparison.iloc[0]['R2']:.4f}")
print(f"   Best MSE: {model_comparison.iloc[0]['MSE']:,.0f}")
print(f"   Best MAE: {model_comparison.iloc[0]['MAE']:,.0f}")

