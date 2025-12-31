import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1. Create folders
# =========================
os.makedirs("visuals", exist_ok=True)
os.makedirs("data", exist_ok=True)

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("data/StudentsPerformance.csv")

# =========================
# 3. Clean Column Names
# =========================
df.columns = df.columns.str.strip().str.replace(" ", "_")

# =========================
# 4. Feature Engineering
# =========================
df["average_score"] = df[
    ["math_score", "reading_score", "writing_score"]
].mean(axis=1)

# =========================
# 5. Save Processed Dataset
# =========================
df.to_csv("data/processed_students_data.csv", index=False)
print("✅ Processed dataset saved")

# =========================
# 6. EDA VISUALIZATIONS
# =========================

# Avg Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["average_score"], kde=True)
plt.title("Average Score Distribution")
plt.savefig("visuals/avg_score_distribution.png")
plt.close()

# Gender vs Avg Score
plt.figure(figsize=(8, 5))
sns.boxplot(x="gender", y="average_score", data=df)
plt.title("Gender vs Average Score")
plt.savefig("visuals/gender_vs_avg_score.png")
plt.close()

# Test Preparation Impact
plt.figure(figsize=(8, 5))
sns.boxplot(x="test_preparation_course", y="average_score", data=df)
plt.title("Test Preparation Course Impact")
plt.savefig("visuals/test_prep_vs_score.png")
plt.close()

print("✅ EDA visuals saved")

# =========================
# 7. ML PREPARATION
# =========================
X = df[
    ["gender", "parental_level_of_education", "lunch", "test_preparation_course"]
]
y = df["average_score"]

X = pd.get_dummies(X, drop_first=True)

# =========================
# 8. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 9. Train Model
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# 10. Prediction & Evaluation
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# =========================
# 11. Save Prediction Plot
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title("Actual vs Predicted Performance")
plt.savefig("visuals/actual_vs_predicted.png")
plt.close()

print("✅ Prediction plot saved")
print("🔥 PROJECT COMPLETED SUCCESSFULLY")
