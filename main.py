import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def main():
  print("--- 1. Loading Dataset ---")
  df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

  print(f"Dataset Shape (Rows, Columns): {df.shape}")
  print(f"Total Data Points (Size): {df.size}")
  print("\nFirst 5 Rows of the Dataset:")
  print(df.head())

  print("\n--- 3. Data Integrity & Types ---")
  missing_values = df.isnull().sum().sum()
  print(f"Total Missing Values: {missing_values}")
  print(f"Data Types Breakdown:\n{df.dtypes.value_counts()}")

  print("\n--- 4. Statistical Summary (Numerical Features) ---")
  print(df.describe())

  # 5. Target Variable Distribution (Class Imbalance Check for Attrition)
  print("\n--- 5. Target Variable (Attrition) Breakdown ---")
  attrition_counts = df["Attrition"].value_counts(normalize=True) * 100
  print(df["Attrition"].value_counts())
  print(f"Percentage Breakdown:\n{attrition_counts}")

  # Plotting Class Imbalance Bar Chart
  plt.figure(figsize=(6, 4))
  sns.countplot(x="Attrition", data=df, palette="Set2")
  plt.title("Target Class Distribution: Employee Attrition")
  plt.xlabel("Attrition")
  plt.ylabel("Count")
  plt.show()

  #Correlation Heatmap for Numerical Features
  print("\n--- 6. Generating Correlation Heatmap ---")
  plt.figure(figsize=(12, 10))
  key_numerical_cols = [
      "Age",
      "MonthlyIncome",
      "TotalWorkingYears",
      "JobSatisfaction",
      "YearsAtCompany",
      "YearsSinceLastPromotion",
      "EnvironmentSatisfaction",
      "JobInvolvement",
  ]
  corr_matrix = df[key_numerical_cols].corr()
  sns.heatmap(
      corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
  )
  plt.title("Correlation Matrix of Key Attrition Predictors")
  plt.show()

  #Box Plots for Outlier Detection
  print("\n--- 7. Generating Box Plots for Outliers ---")
  plt.figure(figsize=(8, 5))
  sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="Pastel1")
  plt.title("Monthly Income Distribution by Attrition Status")
  plt.xlabel("Attrition")
  plt.ylabel("Monthly Income")
  plt.show()

  # Data Preprocessing & Feature Engineering Step
  print("\n--- 8. Data Preprocessing & Feature Engineering ---")

  cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
  df = df.drop(columns=cols_to_drop)
  print(f"Dropped redundant columns: {cols_to_drop}")

  # Feature Engineering
  df["IncomePerYear"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
  df["TenureExperienceRatio"] = df["YearsAtCompany"] / (
      df["TotalWorkingYears"] + 1
  )
  print("Engineered new features: 'IncomePerYear', 'TenureExperienceRatio'")

  binary_cols = ["Attrition", "OverTime", "Gender"]
  le = LabelEncoder()
  for col in binary_cols:
    if col in df.columns:
      df[col] = le.fit_transform(df[col])
  print("Label encoded binary columns (Attrition, OverTime, Gender).")

  nominal_cols = [
      "Department",
      "JobRole",
      "EducationField",
      "BusinessTravel",
      "MaritalStatus",
  ]
  df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
  print(f"One-hot encoded nominal columns. New shape: {df.shape}")

  scaler = StandardScaler()
  scale_cols = [
      "Age",
      "MonthlyIncome",
      "TotalWorkingYears",
      "DistanceFromHome",
      "IncomePerYear",
  ]
  df[scale_cols] = scaler.fit_transform(df[scale_cols])
  print("Standardized continuous numerical features using StandardScaler.")

  # Model Training, Reproducible Split & Cross-Validation Setup 
  print("\n--- 9. Cross-Validation & Model Evaluation ---")

  X = df.drop(columns=["Attrition"])
  y = df["Attrition"]

  # train-test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Calculate dynamic scale_pos_weight for XGBoost based on train set ratio
  neg_count = sum(y_train == 0)
  pos_count = sum(y_train == 1)
  computed_scale_pos_weight = neg_count / pos_count
  print(
      f"Computed XGBoost scale_pos_weight: {computed_scale_pos_weight:.2f}"
  )

  # Initialize all models with fixed random_state and imbalance handling
  models = {
      "Logistic Regression": LogisticRegression(
          max_iter=2000,
          solver="lbfgs",
          class_weight="balanced",
          random_state=42,
      ),
      "Random Forest": RandomForestClassifier(
          class_weight="balanced", random_state=42
      ),
      "XGBoost": XGBClassifier(
          scale_pos_weight=computed_scale_pos_weight,
          eval_metric="logloss",
          random_state=42,
      ),
  }

  # Stratified K-Fold Cross-Validation Setup
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  for name, model in models.items():
    print(f"\n>>> Evaluating Model via 5-Fold Cross-Validation: {name}")

    cv_accuracies = []
    cv_aucs = []

    for train_index, val_index in skf.split(X_train, y_train):
      X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
      y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

      model.fit(X_fold_train, y_fold_train)
      preds = model.predict(X_fold_val)
      probs = model.predict_proba(X_fold_val)[:, 1]

      cv_accuracies.append(accuracy_score(y_fold_val, preds))
      cv_aucs.append(roc_auc_score(y_fold_val, probs))

    print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.4f}")
    print(f"Mean CV ROC-AUC:  {np.mean(cv_aucs):.4f}")

  # Leakage-Free Threshold Tuning Across All Models ---
  print(
      "\n--- 10. Leakage-Free Threshold Tuning & Final Test Evaluation ---"
  )

  tuned_thresholds = {}
  test_predictions = {}

  for name, model in models.items():
    # 1. Get out-of-fold predictions to find the optimal threshold without data leakage
    oof_probs = cross_val_predict(
        model, X_train, y_train, cv=skf, method="predict_proba"
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    tuned_thresholds[name] = optimal_threshold

    print(f"\n{name} Optimal Threshold (from OOF F1): {optimal_threshold:.4f}")

    # 2. Fit on full training data and evaluate on test set using its specific tuned threshold
    model.fit(X_train, y_train)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_tuned = (y_prob_test >= optimal_threshold).astype(int)
    test_predictions[name] = y_pred_tuned

    print(f"--- Test Evaluation for {name} (Threshold = {optimal_threshold:.4f}) ---")
    print(classification_report(y_test, y_pred_tuned))

  # --- VISUALIZATION 1: Side-by-Side Confusion Matrices (Using Tuned Thresholds) ---
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  for idx, (name, model) in enumerate(models.items()):
    y_pred = test_predictions[name]
    thresh = tuned_thresholds[name]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[idx],
        annot_kws={"size": 14},
    )
    axes[idx].set_title(f"{name}\n(Thresh: {thresh:.2f})")
    axes[idx].set_xlabel("Predicted Label")
    axes[idx].set_ylabel("True Label")

  plt.suptitle("Confusion Matrices with Model-Specific Tuned Thresholds", fontsize=16, y=1.03)
  plt.tight_layout()
  plt.show()

  # --- VISUALIZATION 2: Combined ROC Curves ---
  plt.figure(figsize=(8, 6))
  for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

  plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guess")
  plt.xlabel("False Positive Rate", fontsize=12)
  plt.ylabel("True Positive Rate", fontsize=12)
  plt.title(
      "Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=15
  )
  plt.legend(loc="lower right", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.show()

  print(
      "\nPipeline execution, cross-validation, leak-free threshold tuning,"
      " and synchronized visualizations complete!"
  )


if __name__ == "__main__":
  main()