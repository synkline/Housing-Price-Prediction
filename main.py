
import os
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import re


#  Data Acquisition and Cleaning
data_folder = 'archive'
all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

df_list = []
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Combine all data 
data = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")

# Clean column names )
def clean_column(name):
    name = name.strip().lower()
    name = name.replace(' ', '_')
    name = re.sub(r'[()]', '', name)
    return name

data.columns = [clean_column(col) for col in data.columns]

# Drop rows with missing target
if 'price' not in data.columns:
    raise ValueError("❌ 'price' column not found in dataset!")

data = data.dropna(subset=['price'])

# Select features
selected_features = [
    'area',
    'no._of_bedrooms',
    'resale',
    'location'
]

for feat in selected_features:
    if feat not in data.columns:
        # Fallback for datasets where 'no._of_bedrooms' might be different
        if feat == 'no._of_bedrooms' and 'bedrooms' in data.columns:
            selected_features[selected_features.index(feat)] = 'bedrooms'
        else:
            # If the feature is still missing, raise an error
            raise ValueError(f"❌ '{feat}' column not found in dataset!")

X = data[selected_features]
y = data['price']


# --- Preprocessing and Encoding ---

# CRITICAL: Log transform the target (price) to normalize distribution and improve model fit
y_transformed = np.log1p(y)


location_encoder = None
for col in X.select_dtypes(include=['object']).columns:
    if col == 'location':
        # Create and save the encoder for consistent transformation in app_tkinter.py
        location_encoder = LabelEncoder()
        X[col] = location_encoder.fit_transform(X[col])
    else:
        # Use simple LabelEncoder for other object columns (like 'resale')
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split (using y_transformed)
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)


# --- Model Training (Fulfills 'Multiple Models' Capstone Requirement) ---
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# --- Evaluation and Inverse Transform ---
lr_predictions_log = lr_model.predict(X_test)
rf_predictions_log = rf_model.predict(X_test)

# Inverse transform predictions and actual test values for accurate R² score
lr_predictions_real = np.expm1(lr_predictions_log)
rf_predictions_real = np.expm1(rf_predictions_log)
y_test_real = np.expm1(y_test) 

print("\n📊 Model Performance:")
print("Linear Regression R²:", round(r2_score(y_test_real, lr_predictions_real), 3))
print("Random Forest R²:", round(r2_score(y_test_real, rf_predictions_real), 3))


# --- Saving Assets for Deployment ---
os.makedirs('models', exist_ok=True)
joblib.dump(lr_model, 'models/linear_model.joblib')
joblib.dump(rf_model, 'models/random_forest_model.joblib')
joblib.dump(X.columns.tolist(), 'models/model_features.pkl')
joblib.dump(location_encoder, 'models/location_encoder.joblib') # CRITICAL: Saved for Tkinter App

print("\n✅ Models, feature list, and encoder saved successfully!") 