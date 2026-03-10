import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("--- PHASE 9b: TRAINING THE SOVEREIGN BRAIN ---")

# 1. LOAD SUPER DATASET
df = pd.read_csv('SUPER_TRAINING_SET.csv')

# 2. PREP TARGET
# Clean Cases again just to be safe
df['cases'] = pd.to_numeric(df['cases'].astype(str).str.split('/').str[0], errors='coerce').fillna(0).astype(int)

# Encode Disease
le = LabelEncoder()
df['disease_code'] = le.fit_transform(df['disease'].astype(str))
np.save('disease_encoder_v2.npy', le.classes_) # Save new encoder

# 3. LAG FEATURES (Time Travel)
# Sort to ensure lag works
df = df.sort_values(['district', 'year', 'month', 'day'])
df['prev_rain'] = df.groupby('district')['preci'].shift(1).fillna(0)
df['prev_temp'] = df.groupby('district')['temp_celsius'].shift(1).fillna(0)

# 4. SELECT SUPER FEATURES
# Now we include the Holy Grail data
features = [
    'month', 'latitude', 'longitude', 
    'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp',
    'bod', 'fecal_coliform', 'tds', # Water Quality
    'vax_full', 'vax_measles',      # Immunization Shield
    'disease_code'
]
target = 'cases'

print(f"Training on Extended Feature Set: {features}")

X = df[features].fillna(0) # Safety fill
y = df[target]

# 5. TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8)
model.fit(X_train, y_train)

# 6. EVALUATE
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"\n[VICTORY] New Sovereign Brain Trained.")
print(f"Accuracy Metric (MAE): {mae:.2f} cases.")
model.save_model("sovereign_brain.json")
print("Saved 'sovereign_brain.json'.")