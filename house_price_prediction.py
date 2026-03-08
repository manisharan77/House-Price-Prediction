# ============================================================
# House Price Prediction using Machine Learning
# Author: Mani Sharan Bommakanti
# GitHub: github.com/manisharan77
# Description: ML model to predict house prices using
#              EDA, feature engineering, and regression models.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. GENERATE SAMPLE DATASET
# ─────────────────────────────────────────────
def create_sample_dataset(n=500, random_state=42):
    """Generate a realistic house price dataset."""
    np.random.seed(random_state)

    locations = ['Downtown', 'Suburb', 'Rural', 'Uptown', 'Midtown']
    loc_weights = {'Downtown': 1.5, 'Uptown': 1.4, 'Midtown': 1.2,
                   'Suburb': 1.0, 'Rural': 0.7}

    data = {
        'location':        np.random.choice(locations, n),
        'area_sqft':       np.random.randint(500, 5000, n),
        'bedrooms':        np.random.randint(1, 7, n),
        'bathrooms':       np.random.randint(1, 5, n),
        'age_years':       np.random.randint(0, 50, n),
        'garage':          np.random.randint(0, 4, n),
        'garden':          np.random.choice([0, 1], n),
        'floors':          np.random.randint(1, 4, n),
        'distance_city_km':np.random.uniform(1, 50, n).round(1),
        'school_rating':   np.random.randint(1, 11, n),
    }

    df = pd.DataFrame(data)

    # Generate realistic prices
    base_price = 50000
    df['price'] = (
        base_price
        + df['area_sqft']       * 80
        + df['bedrooms']        * 8000
        + df['bathrooms']       * 5000
        - df['age_years']       * 500
        + df['garage']          * 6000
        + df['garden']          * 4000
        + df['floors']          * 3000
        - df['distance_city_km']* 800
        + df['school_rating']   * 2000
        + df['location'].map(loc_weights) * 20000
        + np.random.normal(0, 8000, n)
    ).round(-3)

    df['price'] = df['price'].clip(lower=50000)

    # Introduce some missing values for realism
    for col in ['area_sqft', 'age_years', 'school_rating']:
        mask = np.random.choice([True, False], n, p=[0.03, 0.97])
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
def perform_eda(df):
    print("\n" + "="*60)
    print("       EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    print(f"\n📐 Dataset Shape   : {df.shape}")
    print(f"\n📋 Columns         : {list(df.columns)}")
    print(f"\n📊 Data Types:\n{df.dtypes.to_string()}")
    print(f"\n🔍 Missing Values:\n{df.isnull().sum().to_string()}")
    print(f"\n📈 Statistical Summary:\n{df.describe().round(2).to_string()}")
    print(f"\n💰 Price Distribution:")
    print(f"   Min    : ₹{df['price'].min():,.0f}")
    print(f"   Max    : ₹{df['price'].max():,.0f}")
    print(f"   Mean   : ₹{df['price'].mean():,.0f}")
    print(f"   Median : ₹{df['price'].median():,.0f}")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    """Create new meaningful features."""
    df = df.copy()

    # Price per sqft (only for insight, won't be used as feature)
    df['price_per_sqft'] = (df['price'] / df['area_sqft']).round(2)

    # Room density
    df['room_density'] = df['bedrooms'] + df['bathrooms']

    # House age category
    df['age_category'] = pd.cut(df['age_years'],
                                bins=[-1, 5, 15, 30, 100],
                                labels=['New', 'Recent', 'Old', 'Very Old'])

    # Amenity score
    df['amenity_score'] = df['garage'] + df['garden'] + df['floors']

    # Encode categorical features
    le = LabelEncoder()
    df['location_encoded'] = le.fit_transform(df['location'])
    df['age_category_encoded'] = le.fit_transform(df['age_category'].astype(str))

    print("\n✅ Feature Engineering Complete")
    print(f"   New features added: price_per_sqft, room_density, age_category, amenity_score")

    return df


# ─────────────────────────────────────────────
# 4. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(df):
    """Handle missing values and prepare features."""
    feature_cols = [
        'area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage',
        'garden', 'floors', 'distance_city_km', 'school_rating',
        'location_encoded', 'room_density', 'amenity_score', 'age_category_encoded'
    ]

    X = df[feature_cols].copy()
    y = df['price'].copy()

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    print(f"\n✅ Missing values handled using median imputation")
    print(f"   Features used: {feature_cols}")
    return X_imputed, y, feature_cols


# ─────────────────────────────────────────────
# 5. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols):
    """Train multiple regression models and compare."""
    models = {
        'Linear Regression':      LinearRegression(),
        'Ridge Regression':       Ridge(alpha=1.0),
        'Lasso Regression':       Lasso(alpha=100),
        'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    best_model_name = None
    best_r2 = -999

    print("\n" + "="*60)
    print("        MODEL TRAINING & EVALUATION RESULTS")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    for name, model in models.items():
        # Tree models don't need scaling
        if name in ['Random Forest', 'Gradient Boosting']:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            'model': model, 'y_pred': y_pred,
            'RMSE': rmse, 'MAE': mae, 'R2': r2
        }

        print(f"\n📊 {name}")
        print(f"   R² Score : {r2:.4f} ({r2*100:.2f}%)")
        print(f"   RMSE     : {rmse:,.2f}")
        print(f"   MAE      : {mae:,.2f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} | R² = {best_r2*100:.2f}%")

    # Feature importance for best tree model
    best_tree = 'Random Forest' if results['Random Forest']['R2'] >= results['Gradient Boosting']['R2'] else 'Gradient Boosting'
    importances = results[best_tree]['model'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print(f"\n🔍 Top Feature Importances ({best_tree}):")
    print(feat_imp.head(6).round(4).to_string())

    return results, best_model_name, feat_imp, scaler


# ─────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────
def visualize_results(df, results, best_model_name, feat_imp, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('House Price Prediction — Results Dashboard',
                 fontsize=16, fontweight='bold')

    # Plot 1: Price Distribution
    axes[0, 0].hist(df['price'], bins=30, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('House Price Distribution')
    axes[0, 0].set_xlabel('Price')
    axes[0, 0].set_ylabel('Frequency')

    # Plot 2: Price by Location
    sns.boxplot(x='location', y='price', data=df, ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Price by Location')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # Plot 3: Correlation Heatmap
    num_cols = ['price', 'area_sqft', 'bedrooms', 'bathrooms',
                'age_years', 'garage', 'distance_city_km', 'school_rating']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[0, 2], linewidths=0.5, annot_kws={"size": 7})
    axes[0, 2].set_title('Feature Correlation Heatmap')

    # Plot 4: Model R² Comparison
    model_names = list(results.keys())
    r2_scores = [results[m]['R2'] * 100 for m in model_names]
    short_names = ['Linear\nReg', 'Ridge', 'Lasso', 'Random\nForest', 'Gradient\nBoosting']
    colors = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c']
    bars = axes[1, 0].bar(short_names, r2_scores, color=colors, edgecolor='black')
    axes[1, 0].set_title('Model R² Score Comparison')
    axes[1, 0].set_ylabel('R² Score (%)')
    axes[1, 0].set_ylim(0, 115)
    for bar, r2 in zip(bars, r2_scores):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{r2:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Plot 5: Actual vs Predicted
    y_pred_best = results[best_model_name]['y_pred']
    axes[1, 1].scatter(y_test, y_pred_best, alpha=0.5, color='#3498db')
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[1, 1].set_title(f'Actual vs Predicted — {best_model_name}')
    axes[1, 1].set_xlabel('Actual Price')
    axes[1, 1].set_ylabel('Predicted Price')
    axes[1, 1].legend()

    # Plot 6: Feature Importance
    feat_imp.head(8).plot(kind='barh', ax=axes[1, 2], color='#2ecc71', edgecolor='black')
    axes[1, 2].set_title('Top Feature Importances')
    axes[1, 2].set_xlabel('Importance Score')
    axes[1, 2].invert_yaxis()

    plt.tight_layout()
    plt.savefig('house_price_prediction_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved as 'house_price_prediction_results.png'")
    plt.show()


# ─────────────────────────────────────────────
# 7. PREDICT NEW HOUSES
# ─────────────────────────────────────────────
def predict_new_houses(best_model, scaler, best_model_name):
    """Predict prices for new house inputs."""
    print("\n" + "="*60)
    print("        PREDICTING PRICES FOR NEW HOUSES")
    print("="*60)

    new_houses = pd.DataFrame({
        'area_sqft':          [1200, 3000, 800],
        'bedrooms':           [2,    4,    1  ],
        'bathrooms':          [1,    3,    1  ],
        'age_years':          [5,    20,   35 ],
        'garage':             [1,    2,    0  ],
        'garden':             [0,    1,    0  ],
        'floors':             [1,    2,    1  ],
        'distance_city_km':   [10,   5,    40 ],
        'school_rating':      [7,    9,    5  ],
        'location_encoded':   [0,    2,    4  ],
        'room_density':       [3,    7,    2  ],
        'amenity_score':      [2,    5,    1  ],
        'age_category_encoded':[1,   2,    3  ],
    })

    labels = ['Small Suburban House', 'Large Downtown House', 'Old Rural House']

    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        preds = best_model.predict(new_houses)
    else:
        preds = best_model.predict(scaler.transform(new_houses))

    for label, pred in zip(labels, preds):
        print(f"\n🏠 {label}")
        print(f"   Predicted Price: ${pred:,.0f}")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("   HOUSE PRICE PREDICTION USING MACHINE LEARNING")
    print("   Author: Mani Sharan Bommakanti")
    print("="*60)

    # Create dataset
    df = create_sample_dataset(n=500)
    print(f"\n✅ Dataset created: {df.shape[0]} houses, {df.shape[1]} features")

    # EDA
    perform_eda(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Preprocessing
    X, y, feature_cols = preprocess_data(df)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

    # Train & Evaluate
    results, best_model_name, feat_imp, scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # Visualize
    visualize_results(df, results, best_model_name, feat_imp, y_test)

    # Predict new houses
    predict_new_houses(results[best_model_name]['model'], scaler, best_model_name)

    print("\n✅ House Price Prediction Complete!")
