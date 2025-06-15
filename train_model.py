# 📦 Imports
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
import joblib

# 📥 Load Training Data
df = pd.read_csv("D:/DM Project/train.csv")
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# ➕ Feature Engineering
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2 + 1e-8)
df['HR_per_min'] = df['Heart_Rate'] / (df['Duration'] + 1e-8)

# 🎯 Target and Features
y = df['Calories']
X = df.drop(['id', 'Calories'], axis=1)

# ⚖️ Scale Features
features_to_scale = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
scaler = StandardScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# 💾 Save the scaler
joblib.dump(scaler, "D:/DM Project/scaler.pkl")
print("✅ Scaler saved at D:/DM Project/scaler.pkl")

# 🔁 K-Fold CV with Log Transform
y_log = np.log1p(y)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n📂 Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_true = np.expm1(y_val)

    score = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred))
    print("📏 RMSLE:", score)
    rmsle_scores.append(score)

print("\n📊 Average RMSLE:", np.mean(rmsle_scores))

# 🏁 Train Final Model
final_model = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
final_model.fit(X, y_log)

# 💾 Save model
joblib.dump(final_model, "D:/DM Project/calorie_model.pkl")
print("✅ Model saved at D:/DM Project/calorie_model.pkl")

# 📥 Load Test Data
test_df = pd.read_csv("D:/DM Project/test.csv")
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2 + 1e-8)
test_df['HR_per_min'] = test_df['Heart_Rate'] / (test_df['Duration'] + 1e-8)
test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)

# 🔁 Scale test features
test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

# 🔮 Predict
y_test_pred_log = final_model.predict(test_df)
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.clip(y_test_pred, 0, None)

# 📤 Save submission
submission = pd.DataFrame({'id': test_ids, 'Calories': y_test_pred})
submission.to_csv("D:/DM Project/submission_lgbm_logtransform.csv", index=False)
print("📁 Submission saved at D:/DM Project/submission_lgbm_logtransform.csv")
