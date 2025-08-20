import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ======================
# Step 1: Load dataset
# ======================
df = pd.read_csv("Dataset/matches.csv")   # apna IPL dataset use karna

# ======================
# Step 2: Encode categorical features
# ======================
label_encoders = {}
for col in ["team1", "team2", "venue", "toss_winner", "winner"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ======================
# Step 3: Define features & target
# ======================
X = df[["team1", "team2", "venue", "toss_winner"]]
y = df["winner"]

# ======================
# Step 4: Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Step 5: Train RandomForest
# ======================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ======================
# Step 6: Evaluate Model
# ======================
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
cv_mean = cv_scores.mean() * 100

# Classification report
report = classification_report(y_test, y_pred, zero_division=0)

# ======================
# Step 7: Save model & encoders
# ======================
joblib.dump(rf, "ipl_model.pkl")
joblib.dump(label_encoders["team1"], "team_encoder.pkl")
joblib.dump(label_encoders["venue"], "venue_encoder.pkl")
joblib.dump(label_encoders["toss_winner"], "toss_encoder.pkl")
joblib.dump(label_encoders["winner"], "winner_encoder.pkl")

# ======================
# Step 8: Pretty Print Results
# ======================
print("====================================")
print(f"✅ Accuracy: {accuracy*100:.2f} %")
print("🔄 Cross-Validation :")
for i, score in enumerate(cv_scores, start=1):
    print(f"   Fold {i}: {score*100:.2f} %")
print(f"👉 Average CV Score: {cv_mean:.2f} %")
print("📊 Classification Report:")
print(report)
print("💾 Model & encoders saved successfully!")
print("====================================")
