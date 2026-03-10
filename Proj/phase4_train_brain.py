import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("--- PHASE 4: TRAINING THE EPICLIM PREDICTIVE BRAIN (V2) ---")

# 1. LOAD DATA
# Ensure the filename matches exactly what you have in your folder
df = pd.read_csv('Cleaned_EpiClim_Data.csv')

# 2. DATA CLEANING & ENGINEERING
print("Engineering features...")

# Clean 'Cases' column (Handle '3/1' or simple numbers)
# We coerce errors to NaN, then fill with 0, then convert to Int
df['Cases'] = pd.to_numeric(df['Cases'].astype(str).str.split('/').str[0], errors='coerce').fillna(0).astype(int)

# Fill missing Deaths/Precipitation
df['Deaths'] = df['Deaths'].fillna(0)
df['preci'] = df['preci'].fillna(0)

# 3. ENCODE DISEASE NAMES
# The model needs numbers, not strings like "Malaria". We create a map.
le = LabelEncoder()
df['Disease_Code'] = le.fit_transform(df['Disease'].astype(str))

# Print the mapping so we know what code equals what disease
disease_map = dict(zip(le.transform(le.classes_), le.classes_))
print("Disease Knowledge Map Created.")

# 4. LAG FEATURES (The Time-Travel Logic)
# Sort by District and Date to ensure the "previous week" is actually the previous week
df = df.sort_values(['district', 'year', 'month', 'day'])

# Create "Previous Week Rain" and "Previous Week Temp"
# This teaches the model: "Rain last week = Mosquitoes this week"
df['prev_rain'] = df.groupby('district')['preci'].shift(1).fillna(0)
df['prev_temp'] = df.groupby('district')['Temp_Celsius'].shift(1).fillna(0)

# 5. SELECT FEATURES & TARGET
# Inputs: Where, When, Weather, Disease Type
features = ['month', 'Latitude', 'Longitude', 'preci', 'Temp_Celsius', 'LAI', 'prev_rain', 'prev_temp', 'Disease_Code']
target = 'Cases'

X = df[features]
y = df[target]

# 6. TRAIN PREDICTIVE MODEL
print(f"Training on {len(df)} historical outbreaks...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor (The most powerful gradient boosting tree for this task)
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7)
model.fit(X_train, y_train)

# 7. EVALUATION
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\n[SUCCESS] Model Training Complete.")
print(f"Accuracy Metric (Mean Absolute Error): {mae:.2f} cases.")
print("Interpretation: On average, the model's prediction is off by this many cases.")

# Save the model and the Disease Encoder map for the final Dashboard
model.save_model("epilim_brain.json")
np.save('disease_encoder.npy', le.classes_)
print("Brain saved as 'epilim_brain.json'. Disease Map saved.")