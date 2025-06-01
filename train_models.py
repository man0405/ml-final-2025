import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the dataset"""
    data = pd.read_csv('data/processed_data.csv')
    return data


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, data_type):
    """Evaluate a model and return metrics"""
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        'Model': model_name,
        'Data Type': data_type,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std
    }


def train_and_save_results():
    """Train all models and save results"""
    print("Loading data...")
    data = load_data()

    # Prepare features and target
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Configuration
    test_size = 0.2
    random_state = 42
    pca_variance = 0.95

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    print("Applying PCA...")
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Analyze PCA components
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Choose number of components for selected variance
    n_components_selected = np.argmax(cumulative_variance >= pca_variance) + 1

    # Apply PCA with selected components
    pca_selected = PCA(n_components=n_components_selected)
    X_train_pca_selected = pca_selected.fit_transform(X_train_scaled)
    X_test_pca_selected = pca_selected.transform(X_test_scaled)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf'),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    print("Training models...")
    results_original = []
    results_pca = []

    # Train models on original data
    print("Training on original data...")
    for name, model in models.items():
        print(f"  Training {name}...")
        result = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name, 'Original')
        results_original.append(result)

    # Train models on PCA data
    print("Training on PCA data...")
    for name, model in models.items():
        print(f"  Training {name} on PCA data...")
        result = evaluate_model(
            model, X_train_pca_selected, X_test_pca_selected, y_train, y_test, name, 'PCA')
        results_pca.append(result)

    # Combine results
    all_results = results_original + results_pca
    results_df = pd.DataFrame(all_results)

    # Save results
    print("Saving results...")
    results_df.to_csv('model/training_results.csv', index=False)

    # Save PCA analysis data
    pca_data = {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'n_components_selected': int(n_components_selected),
        'original_features': int(X.shape[1]),
        'pca_variance_threshold': pca_variance,
        'data_info': {
            'total_samples': int(data.shape[0]),
            'features': int(data.shape[1] - 1),
            'target': 'Price'
        }
    }

    with open('model/pca_analysis.json', 'w') as f:
        json.dump(pca_data, f, indent=2)

    # Save feature names
    feature_names = X.columns.tolist()
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print("Training completed!")
    print(f"Results saved to: model/training_results.csv")
    print(f"PCA analysis saved to: model/pca_analysis.json")
    print(f"Feature names saved to: model/feature_names.json")

    # Print summary
    print("\n=== SUMMARY ===")
    best_overall = results_df.loc[results_df['Test R²'].idxmax()]
    print(
        f"Best overall model: {best_overall['Model']} ({best_overall['Data Type']}) - R² = {best_overall['Test R²']:.4f}")

    best_original = results_df[results_df['Data Type'] == 'Original'].loc[
        results_df[results_df['Data Type'] == 'Original']['Test R²'].idxmax()
    ]
    print(
        f"Best on original data: {best_original['Model']} - R² = {best_original['Test R²']:.4f}")

    best_pca = results_df[results_df['Data Type'] == 'PCA'].loc[
        results_df[results_df['Data Type'] == 'PCA']['Test R²'].idxmax()
    ]
    print(
        f"Best on PCA data: {best_pca['Model']} - R² = {best_pca['Test R²']:.4f}")

    reduction_pct = ((X.shape[1] - n_components_selected) / X.shape[1]) * 100
    print(
        f"Dimensionality reduction: {X.shape[1]} → {n_components_selected} features ({reduction_pct:.1f}% reduction)")

    # Save best models
    save_best_models(X_train_scaled, X_test_scaled, X_train_pca_selected, X_test_pca_selected,
                     y_train, y_test, results_df, scaler, pca_selected)


def save_best_models(X_train_scaled, X_test_scaled, X_train_pca_selected, X_test_pca_selected,
                     y_train, y_test, results_df, scaler, pca_selected):
    """Save the best performing models for both original and PCA data"""
    print("\n=== SAVING BEST MODELS ===")

    # Find best models
    best_original = results_df[results_df['Data Type'] == 'Original'].loc[
        results_df[results_df['Data Type'] == 'Original']['Test R²'].idxmax()
    ]

    best_pca = results_df[results_df['Data Type'] == 'PCA'].loc[
        results_df[results_df['Data Type'] == 'PCA']['Test R²'].idxmax()
    ]

    # Define models dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf'),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    # Train and save best model for original data
    best_original_model = models[best_original['Model']]
    best_original_model.fit(X_train_scaled, y_train)

    # Save best original model with all preprocessing components
    best_original_package = {
        'model': best_original_model,
        'scaler': scaler,
        'model_name': best_original['Model'],
        'test_r2': best_original['Test R²'],
        'test_mse': best_original['Test MSE'],
        'data_type': 'Original'
    }

    with open('model/best_original_model.pkl', 'wb') as f:
        pickle.dump(best_original_package, f)

    print(
        f"Best original model saved: {best_original['Model']} (Test R² = {best_original['Test R²']:.4f})")

    # Train and save best model for PCA data
    best_pca_model = models[best_pca['Model']]
    best_pca_model.fit(X_train_pca_selected, y_train)

    # Save best PCA model with all preprocessing components
    best_pca_package = {
        'model': best_pca_model,
        'scaler': scaler,
        'pca': pca_selected,
        'model_name': best_pca['Model'],
        'test_r2': best_pca['Test R²'],
        'test_mse': best_pca['Test MSE'],
        'data_type': 'PCA'
    }

    with open('model/best_pca_model.pkl', 'wb') as f:
        pickle.dump(best_pca_package, f)

    print(
        f"Best PCA model saved: {best_pca['Model']} (Test R² = {best_pca['Test R²']:.4f})")

    # Save model comparison summary
    model_summary = {
        'best_original': {
            'model_name': best_original['Model'],
            'test_r2': float(best_original['Test R²']),
            'test_mse': float(best_original['Test MSE']),
            'test_mae': float(best_original['Test MAE']),
            'cv_r2_mean': float(best_original['CV R² Mean']),
            'cv_r2_std': float(best_original['CV R² Std'])
        },
        'best_pca': {
            'model_name': best_pca['Model'],
            'test_r2': float(best_pca['Test R²']),
            'test_mse': float(best_pca['Test MSE']),
            'test_mae': float(best_pca['Test MAE']),
            'cv_r2_mean': float(best_pca['CV R² Mean']),
            'cv_r2_std': float(best_pca['CV R² Std'])
        },
        'overall_best': {
            'model_name': best_original['Model'] if best_original['Test R²'] > best_pca['Test R²'] else best_pca['Model'],
            'data_type': 'Original' if best_original['Test R²'] > best_pca['Test R²'] else 'PCA',
            'test_r2': float(max(best_original['Test R²'], best_pca['Test R²']))
        }
    }

    with open('model/best_models_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)

    print("Model summary saved to: model/best_models_summary.json")
    print("Best models saved to: model/best_original_model.pkl and model/best_pca_model.pkl")


if __name__ == "__main__":
    train_and_save_results()
