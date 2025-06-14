# Dataset: C:\Users\charitha\Downloads\house_prices.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

print("🏠 House Price Prediction - Kaggle Dataset Analysis")
print("=" * 55)


possible_paths = [
    r"house_prices.csv",  # Same directory as script (PRIMARY)
    r".\house_prices.csv",  # Current directory
    r"C:\Users\charitha\OneDrive\Desktop\projects of my own\house pricing prediction\house_prices.csv",  # Full path
    r"data\house_prices.csv",  # Data subfolder
    r"C:\Users\charitha\Downloads\house_prices.csv"  # Original Downloads location
]

df = None
data_path = None

print(f"\n📊 Attempting to load dataset...")

for path in possible_paths:
    try:
        print(f"Trying: {path}")
        df = pd.read_csv(path)
        data_path = path
        print(f"✅ Dataset loaded successfully from: {path}")
        print(f"Dataset shape: {df.shape}")
        break
    except (FileNotFoundError, PermissionError) as e:
        print(f"❌ Failed to load from {path}: {type(e).__name__}")
        continue

if df is None:
    print("\n❌ Could not load the dataset from any location!")
    print("\n🔧 SOLUTIONS:")
    print("1. Copy 'house_prices.csv' to the same folder as this Python script")
    print("2. Create a 'data' folder and place the CSV file there")
    print("3. Run this command in terminal to copy the file:")
    print("   copy C:\\Users\\charitha\\Downloads\\house_prices.csv .")
    print("4. Or drag and drop the CSV file into your project folder")
    print("\n5. Alternative: Use this code to select file manually:")
    print("   from tkinter import filedialog")
    print("   import tkinter as tk")
    print("   root = tk.Tk()")
    print("   root.withdraw()")
    print("   file_path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])")
    print("   df = pd.read_csv(file_path)")
    
    
    try:
        from tkinter import filedialog
        import tkinter as tk
        print("\n🔄 Opening file dialog...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select house_prices.csv file",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        root.destroy()
        
        if file_path:
            df = pd.read_csv(file_path)
            data_path = file_path
            print(f"✅ Dataset loaded via file dialog: {file_path}")
            print(f"Dataset shape: {df.shape}")
        else:
            print("❌ No file selected")
            exit()
    except ImportError:
        print("❌ tkinter not available for file dialog")
        exit()
    except Exception as e:
        print(f"❌ File dialog failed: {e}")
        exit()


print("\n📋 INITIAL DATA EXPLORATION")
print("=" * 40)

print("Dataset Info:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes.value_counts())

print("\nBasic statistics:")
print(df.describe())


potential_targets = ['SalePrice', 'price', 'Price', 'target', 'y', 'medv', 'median_house_value']
target_col = None

for col in potential_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # If no standard target found, assume last column or ask user
    print("\n⚠️  Target column not automatically detected.")
    print("Available columns:", df.columns.tolist())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", numeric_cols)
    
    # Assume the last numeric column is the target
    if numeric_cols:
        target_col = numeric_cols[-1]
        print(f"🎯 Assuming '{target_col}' is the target variable")
    else:
        print("❌ No numeric columns found for prediction")
        exit()
else:
    print(f"\n🎯 Target variable identified: '{target_col}'")


print("\n🔍 DATA QUALITY ASSESSMENT")
print("=" * 35)


missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percentage': missing_percent.values
}).sort_values('Missing_Percentage', ascending=False)

print("Missing data summary:")
print(missing_df[missing_df['Missing_Count'] > 0].head(10))


if missing_df['Missing_Count'].sum() > 0:
    plt.figure(figsize=(12, 6))
    top_missing = missing_df[missing_df['Missing_Count'] > 0].head(15)
    
    plt.subplot(1, 2, 1)
    plt.barh(top_missing['Column'], top_missing['Missing_Percentage'])
    plt.xlabel('Missing Percentage')
    plt.title('Missing Data by Column')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    
    plt.tight_layout()
    plt.show()


print(f"\n📈 TARGET VARIABLE ANALYSIS: {target_col}")
print("=" * 45)

target_stats = df[target_col].describe()
print("Target variable statistics:")
print(target_stats)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.boxplot(df[target_col])
plt.title(f'{target_col} Boxplot')
plt.ylabel(target_col)

plt.subplot(1, 3, 3)

log_target = np.log1p(df[target_col])
plt.hist(log_target, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title(f'Log-transformed {target_col}')
plt.xlabel(f'log({target_col})')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


from scipy.stats import skew
target_skewness = skew(df[target_col].dropna())
print(f"Target skewness: {target_skewness:.3f}")
if abs(target_skewness) > 1:
    print("⚠️  High skewness detected. Consider log transformation.")


print("\n⚙️ DATA PREPROCESSING")
print("=" * 30)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:10]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")


if target_col in numeric_cols:
    numeric_cols.remove(target_col)


print(f"\nHandling missing values...")


if numeric_cols:
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])


if categorical_cols:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

print("✅ Missing values handled")


if categorical_cols:
    print(f"Encoding categorical variables...")
    
    
    label_encoders = {}
    for col in categorical_cols:
        if df[col].nunique() < 50:  # Only encode if less than 50 unique values
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            numeric_cols.append(col + '_encoded')
    
    print(f"✅ Encoded {len(label_encoders)} categorical variables")


print("\n🎯 FEATURE SELECTION")
print("=" * 25)


features = numeric_cols.copy()


features_to_remove = []
for col in features:
    if col in df.columns:
        # Check variance
        if df[col].var() == 0:
            features_to_remove.append(col)
        # Check for ID-like columns
        if 'id' in col.lower() or 'Id' in col:
            features_to_remove.append(col)

features = [f for f in features if f not in features_to_remove]
print(f"Selected {len(features)} features for modeling")


if len(features) > 1:
    feature_data = df[features + [target_col]]
    correlation_matrix = feature_data.corr()
  
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Top correlations with target
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with {target_col}:")
    print(target_corr[1:11])  # Exclude target itself
    
    # Select top correlated features
    top_features = target_corr[1:21].index.tolist()  # Top 20 features
    features = [f for f in top_features if f in features]

print(f"Final feature set: {len(features)} features")


X = df[features]
y = df[target_col]


X = X.fillna(X.median())
y = y.fillna(y.median())

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n🤖 MODEL TRAINING")
print("=" * 25)


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}


results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    try:
        
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"✅ {name} completed:")
        print(f"   RMSE: {rmse:,.2f}")
        print(f"   MAE: {mae:,.2f}")
        print(f"   R² Score: {r2:.4f}")
        
    except Exception as e:
        print(f"❌ Error training {name}: {str(e)}")


print("\n📊 MODEL EVALUATION")
print("=" * 25)

if results:
    
    performance_data = []
    for name, metrics in results.items():
        performance_data.append({
            'Model': name,
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R² Score': metrics['r2']
        })
    
    performance_df = pd.DataFrame(performance_data)
    print("\n📈 Model Performance Comparison:")
    print(performance_df.to_string(index=False, float_format='%.4f'))
    
    
    best_model_idx = performance_df['R² Score'].idxmax()
    best_model_name = performance_df.loc[best_model_idx, 'Model']
    best_r2 = performance_df.loc[best_model_idx, 'R² Score']
    
    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
    
   
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    
    axes[0, 0].bar(performance_df['Model'], performance_df['RMSE'], color='skyblue')
    axes[0, 0].set_title('RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(performance_df['Model'], performance_df['MAE'], color='lightcoral')
    axes[0, 1].set_title('MAE Comparison')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(performance_df['Model'], performance_df['R² Score'], color='lightgreen')
    axes[1, 0].set_title('R² Score Comparison')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
   
    best_pred = results[best_model_name]['predictions']
    axes[1, 1].scatter(y_test, best_pred, alpha=0.6, color='purple')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Prices')
    axes[1, 1].set_ylabel('Predicted Prices')
    axes[1, 1].set_title(f'Actual vs Predicted - {best_model_name}')
    
    plt.tight_layout()
    plt.show()
    
    
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        best_model = results[best_model_name]['model']
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🌟 Top 15 Important Features ({best_model_name}):")
        print(feature_importance.head(15).to_string(index=False, float_format='%.4f'))
        
       
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    
    print(f"\n🔍 RESIDUAL ANALYSIS - {best_model_name}")
    print("=" * 35)
    
    residuals = y_test - best_pred
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(best_pred, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    
    print(f"\n🎯 SAMPLE PREDICTIONS - {best_model_name}")
    print("=" * 35)
    
    sample_predictions = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': best_pred[:10],
        'Difference': (y_test.values[:10] - best_pred[:10]),
        'Abs_Error': np.abs(y_test.values[:10] - best_pred[:10])
    })
    
    print(sample_predictions.to_string(index=False, float_format='%.2f'))
    
    
    print(f"\n📋 FINAL SUMMARY")
    print("=" * 20)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 R² Score: {best_r2:.4f} ({best_r2*100:.1f}% variance explained)")
    print(f"💰 Average Prediction Error: {results[best_model_name]['mae']:,.2f}")
    print(f"🎯 Root Mean Square Error: {results[best_model_name]['rmse']:,.2f}")
    
    print(f"\n💡 Model Performance Interpretation:")
    if best_r2 > 0.8:
        print("✅ Excellent model performance!")
    elif best_r2 > 0.6:
        print("✅ Good model performance")
    elif best_r2 > 0.4:
        print("⚠️  Moderate model performance")
    else:
        print("❌ Poor model performance - consider feature engineering")
    
    print(f"\n🚀 Next Steps:")
    print("• Hyperparameter tuning for better performance")
    print("• Feature engineering (polynomial features, interactions)")
    print("• Cross-validation for robust evaluation")
    print("• Try advanced models (XGBoost, Neural Networks)")
    print("• Ensemble methods for improved accuracy")
    
    print(f"\n💾 Model ready for deployment!")

else:
    print("❌ No models were successfully trained. Check your data and features.")
