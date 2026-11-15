# =====================================================================
#  EARLY STROKE PREDICTION - COMPLETE DATA SCIENCE PIPELINE
#  Each "# %%" block is one cell. In VS Code click "Run Cell" above it,
#  or run the whole file with:  python stroke_pipeline.py
#  Every stage saves its outputs (tables + charts) into  outputs/
# =====================================================================

# %% ---------------- STAGE 0: SETUP ----------------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_style("whitegrid")


def save_fig(name):
    """Save the current chart into outputs/ and show it."""
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.show()


# %% ---------------- STAGE 1: PROBLEM DEFINITION ----------------
problem = """
PROBLEM STATEMENT
-----------------
Goal      : Predict whether a patient is likely to have a stroke.
Type      : Binary classification (0 = No stroke, 1 = Stroke)
Input     : Age, gender, hypertension, heart disease, marital status,
            work type, residence, glucose level, BMI, smoking status
Target    : stroke
Challenge : Very few stroke cases (imbalanced data)
Success   : High RECALL (missing a real stroke is worse than a false alarm),
            supported by F1-score and ROC-AUC
"""
print(problem)
with open(f"{OUT}/stage1_problem_definition.txt", "w", encoding="utf-8") as f:
    f.write(problem)


# %% ---------------- STAGE 2: DATA COLLECTION ----------------
df = pd.read_csv("healthcare-stroke-dataset.csv")

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
df.head(10).to_csv(f"{OUT}/stage2_sample_data.csv", index=False)


# %% ---------------- STAGE 3: DATA UNDERSTANDING ----------------
print("DATA TYPES & NON-NULL COUNTS")
df.info()

print("\nSTATISTICAL SUMMARY (numeric)")
summary = df.describe().T
print(summary)
summary.to_csv(f"{OUT}/stage3_numeric_summary.csv")

print("\nTARGET DISTRIBUTION")
target_counts = df["stroke"].value_counts()
print(target_counts)
print((df["stroke"].value_counts(normalize=True) * 100).round(2).astype(str) + " %")

plt.figure(figsize=(6, 4))
sns.countplot(x="stroke", data=df, palette=["#4CAF50", "#F44336"])
plt.title("Stroke vs No Stroke (Class Imbalance)")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
save_fig("stage3_target_distribution")


# %% ---------------- STAGE 4: DATA CLEANING ----------------
# 4.1 Missing values
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])
missing.to_csv(f"{OUT}/stage4_missing_values.csv")

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
save_fig("stage4_missing_heatmap")

# 4.2 Duplicates (checked after dropping 'id', since ids are unique per row)
df = df.drop(columns=["id"])
print("\nDuplicate rows:", df.duplicated().sum())
# Identical rows can land in BOTH train and test and inflate scores, so remove them
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(df))

# 4.3 'id' was dropped above: it is only an identifier with no predictive value

# 4.4 Rare / invalid category: gender 'Other' has only 1 row
print("\nGender counts:\n", df["gender"].value_counts())
df = df[df["gender"] != "Other"].reset_index(drop=True)

# 4.5 Outlier check (just inspect, medical values are real so we keep them)
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1); sns.boxplot(x=df["bmi"]);               plt.title("BMI")
plt.subplot(1, 2, 2); sns.boxplot(x=df["avg_glucose_level"]); plt.title("Glucose")
save_fig("stage4_outliers_boxplot")

print("\nShape after cleaning:", df.shape)
# NOTE: BMI missing values are filled in STAGE 6 AFTER the train/test split,
# using the training median only (avoids data leakage).


# %% ---------------- STAGE 5: EXPLORATORY DATA ANALYSIS ----------------
# 5.1 Numeric distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["age", "avg_glucose_level", "bmi"]):
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"{col} distribution")
save_fig("stage5_numeric_distributions")

# 5.2 Categorical distributions
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), cat_cols):
    sns.countplot(x=col, data=df, ax=ax)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
axes.ravel()[-1].axis("off")
save_fig("stage5_categorical_distributions")

# 5.3 Risk factors vs stroke
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(x="stroke", y="age", data=df, ax=axes[0]);               axes[0].set_title("Age vs Stroke")
sns.boxplot(x="stroke", y="avg_glucose_level", data=df, ax=axes[1]); axes[1].set_title("Glucose vs Stroke")
sns.boxplot(x="stroke", y="bmi", data=df, ax=axes[2]);               axes[2].set_title("BMI vs Stroke")
save_fig("stage5_numeric_vs_stroke")

# 5.4 Stroke rate (%) per category - fairer than raw counts
rate_cols = ["hypertension", "heart_disease", "smoking_status", "work_type", "Residence_type"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), rate_cols):
    (df.groupby(col)["stroke"].mean() * 100).plot(kind="bar", ax=ax)
    ax.set_ylabel("Stroke rate (%)")
    ax.set_title(f"Stroke rate by {col}")
    ax.tick_params(axis="x", rotation=30)
axes.ravel()[-1].axis("off")
save_fig("stage5_stroke_rate_by_category")

# 5.5 Correlation heatmap (numeric only)
plt.figure(figsize=(7, 5))
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
save_fig("stage5_correlation_heatmap")
corr.to_csv(f"{OUT}/stage5_correlation_table.csv")


# %% ---------------- STAGE 6: DATA PREPROCESSING ----------------
# 6.1 One-hot encode categorical columns
df_enc = pd.get_dummies(
    df, columns=["gender", "ever_married", "Residence_type",
                 "work_type", "smoking_status"], drop_first=True)

# 6.2 Separate features and target
X = df_enc.drop("stroke", axis=1)
y = df_enc["stroke"]

# 6.3 Stratified train / test split (keeps the stroke ratio equal in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 6.4 Fill missing BMI using the TRAINING median only
bmi_median = X_train["bmi"].median()
X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
print("Missing values left - train:", X_train.isnull().sum().sum(),
      "| test:", X_test.isnull().sum().sum())

# 6.5 Scaling happens inside the model Pipeline in Stage 7 (fit on train only)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)
print("Stroke share  train: %.2f%%  test: %.2f%%" %
      (y_train.mean() * 100, y_test.mean() * 100))
print("\nFinal feature list:", list(X.columns))

pd.DataFrame({"Set": ["Train", "Test"],
              "Rows": [len(X_train), len(X_test)],
              "Stroke cases": [int(y_train.sum()), int(y_test.sum())]}
             ).to_csv(f"{OUT}/stage6_split_summary.csv", index=False)
pd.Series(X.columns, name="feature").to_csv(f"{OUT}/stage6_feature_list.csv", index=False)


# %% ---------------- STAGE 7: MODEL BUILDING & TRAINING ----------------
# class_weight="balanced" handles the imbalance by penalising missed strokes more
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

rf_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced", min_samples_leaf=5))])

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print("Both models trained successfully.")

models = {"Logistic Regression": lr_model, "Random Forest": rf_model}
pd.DataFrame({"Model": list(models),
              "Algorithm": ["LogisticRegression", "RandomForestClassifier"],
              "Training rows": [len(X_train)] * 2}
             ).to_csv(f"{OUT}/stage7_models_trained.csv", index=False)


# %% ---------------- STAGE 8: MODEL EVALUATION & VALIDATION ----------------
rows = []
for name, m in models.items():
    pred = m.predict(X_test)
    prob = m.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)})
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, zero_division=0))

metrics_df = pd.DataFrame(rows).round(3)
print("\nTEST SET COMPARISON\n", metrics_df.to_string(index=False))
metrics_df.to_csv(f"{OUT}/stage8_test_metrics.csv", index=False)

metrics_df.set_index("Model").plot(kind="bar", figsize=(9, 5))
plt.title("Model Performance Comparison (Test Set)")
plt.ylim(0, 1); plt.xticks(rotation=0)
plt.legend(loc="lower right")
save_fig("stage8_model_comparison")

# 8.1 Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, m) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, m.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No stroke", "Stroke"], yticklabels=["No stroke", "Stroke"])
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
save_fig("stage8_confusion_matrices")

# 8.2 ROC curves
plt.figure(figsize=(6, 5))
for name, m in models.items():
    fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves"); plt.legend()
save_fig("stage8_roc_curves")

# 8.3 Train vs Test (overfitting check)
fit_rows = []
for name, m in models.items():
    fit_rows.append({
        "Model": name,
        "Train Recall": recall_score(y_train, m.predict(X_train)),
        "Test Recall": recall_score(y_test, m.predict(X_test)),
        "Train F1": f1_score(y_train, m.predict(X_train)),
        "Test F1": f1_score(y_test, m.predict(X_test))})
fit_df = pd.DataFrame(fit_rows).round(3)
print("\nTRAIN vs TEST\n", fit_df.to_string(index=False))
fit_df.to_csv(f"{OUT}/stage8_train_vs_test.csv", index=False)

fit_df.set_index("Model").plot(kind="bar", figsize=(9, 5))
plt.title("Train vs Test (large gap = overfitting)")
plt.ylim(0, 1); plt.xticks(rotation=0)
save_fig("stage8_train_vs_test")

# 8.4 5-fold stratified cross-validation (validation on the training data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, m in models.items():
    cv = cross_validate(m, X_train, y_train, cv=skf,
                        scoring=["recall", "f1", "roc_auc"])
    cv_rows.append({"Model": name,
                    "CV Recall (mean)": cv["test_recall"].mean(),
                    "CV Recall (std)": cv["test_recall"].std(),
                    "CV F1 (mean)": cv["test_f1"].mean(),
                    "CV ROC-AUC (mean)": cv["test_roc_auc"].mean()})
cv_df = pd.DataFrame(cv_rows).round(3)
print("\n5-FOLD CROSS-VALIDATION\n", cv_df.to_string(index=False))
cv_df.to_csv(f"{OUT}/stage8_cross_validation.csv", index=False)


# %% ---------------- STAGE 9: DEPLOYMENT & PRODUCTION ----------------
# 9.1 Pick the best model by recall (our success metric), tie-break on ROC-AUC
best_name = metrics_df.sort_values(["Recall", "ROC-AUC"], ascending=False).iloc[0]["Model"]
best_model = models[best_name]
print("Best model selected:", best_name)

# 9.2 Save everything needed to predict on new patients
artifact = {"model": best_model,
            "feature_columns": list(X.columns),
            "bmi_median": float(bmi_median),
            "model_name": best_name}
joblib.dump(artifact, f"{OUT}/stroke_model.joblib")
print("Saved ->", f"{OUT}/stroke_model.joblib")


# 9.3 Reusable prediction function (this is what an app / API would call)
def predict_stroke(patient: dict, threshold: float = 0.5):
    art = joblib.load(f"{OUT}/stroke_model.joblib")
    row = pd.DataFrame([patient])
    row = pd.get_dummies(row)
    row = row.reindex(columns=art["feature_columns"], fill_value=0)
    row["bmi"] = row["bmi"].fillna(art["bmi_median"])
    prob = art["model"].predict_proba(row)[0, 1]
    return {"stroke_probability": round(float(prob), 3),
            "prediction": "HIGH RISK" if prob >= threshold else "LOW RISK"}


# 9.4 Test it with two sample patients (raw, un-encoded values)
patients = {
    "Patient A (older, hypertensive)": {
        "gender": "Male", "age": 72, "hypertension": 1, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 210.0, "bmi": 32.0, "smoking_status": "formerly smoked"},
    "Patient B (young, healthy)": {
        "gender": "Female", "age": 25, "hypertension": 0, "heart_disease": 0,
        "ever_married": "No", "work_type": "Private", "Residence_type": "Rural",
        "avg_glucose_level": 85.0, "bmi": 22.0, "smoking_status": "never smoked"}}

results = []
for label, p in patients.items():
    r = predict_stroke(p)
    print(label, "->", r)
    results.append({"Patient": label, **r})
pd.DataFrame(results).to_csv(f"{OUT}/stage9_sample_predictions.csv", index=False)

print("\nAll stages complete. Check the 'outputs' folder.")