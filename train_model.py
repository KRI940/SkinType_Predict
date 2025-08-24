import os
os.chdir("K:\\Kritika\\Internship Project\\AI Skin type Predictor\\skin\\skin_flask_project\\skin_flask")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load and preprocess data
df = pd.read_csv("skin_data_100k.csv")

# Identify categorical columns
categorical_cols = ["gender", "weather", "oiliness", "acne", 
                    "tightness_after_wash", "makeup_usage", 
                    "flaking", "redness_itchiness"]

# Create encoders for each categorical column
encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(df["skin_type"])

# Separate features and target
X = df.drop("skin_type", axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Train RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Save model, scaler, and encoders
joblib.dump(model, "skin_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
for name, encoder in encoders.items():
    joblib.dump(encoder, f"{name}_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

# Print scores
print("RandomForest score:", model.score(X_test,y_test))
print("XGBoost score:", xgb.score(X_test,y_test))

print(pd.Series(model.predict(X_test)).value_counts())
