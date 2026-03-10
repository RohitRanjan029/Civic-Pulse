import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

print("--- PHASE 9b: PURE CAUSAL TRAINING (NO GEO-CHEATING) ---")

df = pd.read_csv('SUPER_TRAINING_SET.csv')
df['cases'] = pd.to_numeric(df['cases'].astype(str).str.split('/').str[0], errors='coerce').fillna(0).astype(int)

le = LabelEncoder()
df['disease_code'] = le.fit_transform(df['disease'].astype(str))
np.save('disease_encoder_v2.npy', le.classes_)

df = df.sort_values(['district', 'year', 'month', 'day'])
df['prev_rain'] = df.groupby('district')['preci'].shift(1).fillna(0)
df['prev_temp'] = df.groupby('district')['temp_celsius'].shift(1).fillna(0)

# ML FIX: Removed Latitude and Longitude. 
# The AI must now rely entirely on environment and biology.
features = [
    'month', 'preci', 'temp_celsius', 'lai', 'prev_rain', 'prev_temp',
    'bod', 'fecal_coliform', 'tds', 
    'vax_full', 'vax_measles', 'disease_code'
]
target = 'cases'

df = df.sort_values(by=['year', 'month', 'day'])
X = df[features].fillna(0)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Added Subsample and Colsample to prevent overfitting
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

importance = model.feature_importances_
feature_weights = sorted(dict(zip(features, importance)).items(), key=lambda x: x[1], reverse=True)

print(f"\n[VICTORY] Pure Causal Brain Trained.")
print(f"Accuracy Metric (MAE): {mae:.2f} cases.")
print("\n--- WHAT THE AI LEARNED (CAUSAL DRIVERS) ---")
for feat, weight in feature_weights[:5]:
    print(f" - {feat}: {weight*100:.1f}% influence")

model.save_model("sovereign_brain_perfected.json")