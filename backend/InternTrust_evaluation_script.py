"""
InternTrust Model Evaluation
--------------------------------
Evaluates the actual Logistic Regression fraud-detection model
using the SAME feature-extraction logic used by RiskEngine.

Evaluation method:
- dataset: Fake_Real_Job_Posting_Cleaned.csv
- 5-Fold Stratified Cross-Validation
- Logistic Regression(max_iter=1000)
- Real all-MiniLM-L6-v2 embedding when available
- Same TextAnalyzer and UrgencyDetector used by the project

IMPORTANT:
This script does NOT modify the dataset or labels.
"""

import sys
import csv
import json
from pathlib import Path

# ============================================================
# 1. PROJECT PATH
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from services.text_analyzer import TextAnalyzer
from services.urgency_detector import UrgencyDetector

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 2. DATASET PATH
# ============================================================

CSV_PATH = BACKEND_DIR / "data" / "Fake_Real_Job_Posting_Cleaned.csv"

print("=" * 60)
print("INTERNTRUST MODEL EVALUATION")
print("=" * 60)

print("\nDataset:")
print(CSV_PATH)

if not CSV_PATH.exists():
    print("\nERROR:  CSV file was not found!")
    print("Expected location:")
    print(CSV_PATH)
    sys.exit(1)

# ============================================================
# 3. LOAD DATASET
# ============================================================

rows = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    print("\nCSV columns:")
    print(reader.fieldnames)

    for row in reader:

        fraud = str(row.get("fraudulent", "")).strip().lower()

        # Real = 0
        if fraud in {
            "real",
            "0",
            "false",
            "legitimate"
        }:
            target = 0

        # Fake = 1
        elif fraud in {
            "fake",
            "1",
            "true",
            "fraud",
            "fraudulent",
            "scam"
        }:
            target = 1

        else:
            print("WARNING: Unknown label skipped:", fraud)
            continue

        title_val = row.get("title", "") or ""

        # Same fallback logic used in evaluation script
        text_val = (
            row.get("description")
            or row.get("requirements")
            or ""
        )

        rows.append(
            (
                title_val,
                text_val,
                target
            )
        )

print("\nDataset loaded successfully.")

total = len(rows)
fake_count = sum(r[2] == 1 for r in rows)
real_count = sum(r[2] == 0 for r in rows)

print("Total records :", total)
print("Real records  :", real_count)
print("Fake records  :", fake_count)

# ============================================================
# 4. CHECK DATASET
# ============================================================

if total < 5:
    print("\nERROR: Not enough records for 5-fold cross-validation.")
    sys.exit(1)

if fake_count < 5:
    print("\nWARNING:")
    print(
        "There are fewer than 5 Fake records, so 5-fold "
        "stratified cross-validation is not possible."
    )
    print("Fake records:", fake_count)
    sys.exit(1)

# ============================================================
# 5. LOAD REAL PROJECT NLP COMPONENTS
# ============================================================

print("\nInitializing TextAnalyzer...")
print(
    "This will try to load "
    "all-MiniLM-L6-v2."
)

text_analyzer = TextAnalyzer()

print("\nInitializing UrgencyDetector...")

urgency_detector = UrgencyDetector()

# Check which embedding is actually being used
model_loader = text_analyzer.model_loader

using_real_transformer = (
    model_loader.model is not None
)

print("\n------------------------------------------------------------")
print(
    "Using real all-MiniLM-L6-v2:",
    using_real_transformer
)
print("------------------------------------------------------------")

if not using_real_transformer:

    print("\nWARNING:")
    print(
        "The real all-MiniLM-L6-v2 model was NOT loaded."
    )
    print(
        "The project's fallback embedding is being used."
    )
    print(
        "For your final paper, run this script after "
        "confirming the real transformer is loaded."
    )

# ============================================================
# 6. SAME FEATURE EXTRACTION LOGIC
# ============================================================

def extract_features(title, text):

    """
    Same feature extraction logic used by the project.

    Features:
    1. Text/scam-pattern similarity score
    2. Urgency score
    3. Text length score
    """

    combined_text = (
        f"{title or ''} {text or ''}"
    ).strip()

    # Text analysis
    try:
        text_score = float(
            text_analyzer.analyze(combined_text)
        )
    except Exception as e:
        print(
            "Text analysis warning:",
            e
        )
        text_score = 0.0

    # Urgency analysis
    try:
        urgency_score = float(
            urgency_detector.analyze(text or "")
        )
    except Exception as e:
        print(
            "Urgency analysis warning:",
            e
        )
        urgency_score = 0.0

    # Length score
    length_score = min(
        len(combined_text) / 5.0,
        100.0
    )

    return [
        text_score,
        urgency_score,
        length_score
    ]


# ============================================================
# 7. EXTRACT FEATURES
# ============================================================

print("\nExtracting features from all records...")

X = []
y = []

for title_val, text_val, target in rows:

    features = extract_features(
        title_val,
        text_val
    )

    X.append(features)
    y.append(target)

print("Feature extraction completed.")

print("\nFeature vector example:")
print(X[0])

# ============================================================
# 8. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

print("\n============================================================")
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("============================================================")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

print("\nTraining Logistic Regression...")
print("Model: LogisticRegression(max_iter=1000)")
print("Folds: 5")
print("Shuffle: True")
print("Random state: 42")


# ============================================================
# 8A. INDIVIDUAL FOLD METRICS
# ============================================================

print("\n============================================================")
print("INDIVIDUAL FOLD RESULTS")
print("============================================================")

fold_results = []

# Store feature importance from each fold
fold_feature_importance = []

# Feature names corresponding to X columns
feature_names = [
    "Semantic Similarity",
    "Urgency Score",
    "Length Feature"
]

for fold_number, (train_index, test_index) in enumerate(
    cv.split(X, y),
    start=1
):

    # Split data according to current fold
    X_train = [X[i] for i in train_index]
    X_test = [X[i] for i in test_index]

    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]

    # Train a fresh Logistic Regression model
    fold_model = LogisticRegression(
        max_iter=1000
    )

    fold_model.fit(
        X_train,
        y_train
    )

    # Predict on held-out fold
    fold_pred = fold_model.predict(
        X_test
    )

    # Calculate fold metrics
    fold_accuracy = accuracy_score(
        y_test,
        fold_pred
    )

    fold_precision = precision_score(
        y_test,
        fold_pred,
        zero_division=0
    )

    fold_recall = recall_score(
        y_test,
        fold_pred,
        zero_division=0
    )

    fold_f1 = f1_score(
        y_test,
        fold_pred,
        zero_division=0
    )

    fold_results.append({
        "Fold": fold_number,
        "Accuracy": float(fold_accuracy),
        "Precision": float(fold_precision),
        "Recall": float(fold_recall),
        "F1-score": float(fold_f1)
    })

    print("\nFold", fold_number)
    print("------------------------------")
    print(
        f"Accuracy :  {fold_accuracy * 100:.2f}%"
    )
    print(
        f"Precision:  {fold_precision * 100:.2f}%"
    )
    print(
        f"Recall   :  {fold_recall * 100:.2f}%"
    )
    print(
        f"F1-score :  {fold_f1 * 100:.2f}%"
    )

    # ========================================================
    # FEATURE IMPORTANCE FOR THIS FOLD
    # ========================================================
    #
    # Permutation importance measures how much the F1-score
    # changes when each feature is shuffled.
    #
    # This is calculated only on the held-out fold so that
    # feature importance is not based on the training data.
    # ========================================================

    import numpy as np
    from sklearn.inspection import permutation_importance

    X_test_array = np.array(X_test)
    y_test_array = np.array(y_test)

    permutation_result = permutation_importance(
        fold_model,
        X_test_array,
        y_test_array,
        scoring="f1",
        n_repeats=20,
        random_state=42
    )

    fold_importance = permutation_result.importances_mean

    fold_feature_importance.append(
        fold_importance
    )


# ============================================================
# 8B. FOLD METRIC SUMMARY
# ============================================================

print("\n============================================================")
print("5-FOLD METRIC SUMMARY")
print("============================================================")

import numpy as np

accuracy_values = np.array([
    result["Accuracy"]
    for result in fold_results
])

precision_values = np.array([
    result["Precision"]
    for result in fold_results
])

recall_values = np.array([
    result["Recall"]
    for result in fold_results
])

f1_values = np.array([
    result["F1-score"]
    for result in fold_results
])


accuracy_mean = accuracy_values.mean()
accuracy_std = accuracy_values.std(ddof=1)

precision_mean = precision_values.mean()
precision_std = precision_values.std(ddof=1)

recall_mean = recall_values.mean()
recall_std = recall_values.std(ddof=1)

f1_mean = f1_values.mean()
f1_std = f1_values.std(ddof=1)


print(
    f"\nAccuracy :  "
    f"{accuracy_mean * 100:.2f}% "
    f"+/- {accuracy_std * 100:.2f}%"
)

print(
    f"Precision:  "
    f"{precision_mean * 100:.2f}% "
    f"+/- {precision_std * 100:.2f}%"
)

print(
    f"Recall   :  "
    f"{recall_mean * 100:.2f}% "
    f"+/- {recall_std * 100:.2f}%"
)

print(
    f"F1-score :  "
    f"{f1_mean * 100:.2f}% "
    f"+/- {f1_std * 100:.2f}%"
)


# ============================================================
# 8C. FEATURE IMPORTANCE SUMMARY
# ============================================================

print("\n============================================================")
print("FEATURE IMPORTANCE")
print("============================================================")

# Convert fold importance values into NumPy array
feature_importance_array = np.array(
    fold_feature_importance
)

# Average importance across all five folds
mean_feature_importance = (
    feature_importance_array.mean(axis=0)
)

std_feature_importance = (
    feature_importance_array.std(
        axis=0,
        ddof=1
    )
)

print(
    "\nMean permutation importance "
    "(F1-score decrease):"
)

feature_importance_results = []

for i, feature_name in enumerate(
    feature_names
):

    mean_value = float(
        mean_feature_importance[i]
    )

    std_value = float(
        std_feature_importance[i]
    )

    feature_importance_results.append({
        "Feature": feature_name,
        "Mean Permutation Importance": mean_value,
        "Std": std_value
    })

    print(
        f"{feature_name}: "
        f"{mean_value:.6f} "
        f"+/- {std_value:.6f}"
    )


# ============================================================
# 9. OVERALL OUT-OF-FOLD PREDICTIONS
# ============================================================

print("\n============================================================")
print("FINAL OUT-OF-FOLD EVALUATION")
print("============================================================")

# Generate one held-out prediction for every record.
# This preserves your original evaluation method.

y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)


# ============================================================
# 10. CALCULATE OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# 11. PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy :  {accuracy * 100:.2f}%"
)

print(
    f"Precision:  {precision * 100:.2f}%"
)

print(
    f"Recall   :  {recall * 100:.2f}%"
)

print(
    f"F1-score :  {f1 * 100:.2f}%"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n------------------------------------------------------------")
print("CONFUSION MATRIX")
print("------------------------------------------------------------")

print("[[TN FP]")
print(" [FN TP]]")

print(cm)

print("\n------------------------------------------------------------")
print("CONFUSION MATRIX VALUES")
print("------------------------------------------------------------")

print(
    "True Negative  (TN):",
    tn
)

print(
    "False Positive (FP):",
    fp
)

print(
    "False Negative (FN):",
    fn
)

print(
    "True Positive  (TP):",
    tp
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n------------------------------------------------------------")
print("CLASSIFICATION REPORT")
print("------------------------------------------------------------")

classification_results = classification_report(
    y,
    y_pred,
    target_names=[
        "Real",
        "Fake"
    ],
    zero_division=0
)

print(
    classification_results
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = {

    "dataset": str(CSV_PATH),

    "total_records": total,

    "real_records": real_count,

    "fake_records": fake_count,

    "evaluation_method":
        "5-Fold Stratified Cross-Validation",

    "random_state": 42,

    "model":
        "LogisticRegression(max_iter=1000)",

    "embedding":
        "all-MiniLM-L6-v2"
        if using_real_transformer
        else "Project fallback embedding",

    "using_real_transformer":
        using_real_transformer,

    # --------------------------------------------------------
    # Overall out-of-fold metrics
    # --------------------------------------------------------

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    # --------------------------------------------------------
    # Individual fold metrics
    # --------------------------------------------------------

    "fold_results":
        fold_results,

    "fold_mean_std": {

        "accuracy_mean":
            float(accuracy_mean),

        "accuracy_std":
            float(accuracy_std),

        "precision_mean":
            float(precision_mean),

        "precision_std":
            float(precision_std),

        "recall_mean":
            float(recall_mean),

        "recall_std":
            float(recall_std),

        "f1_mean":
            float(f1_mean),

        "f1_std":
            float(f1_std)
    },

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    "feature_importance":
        feature_importance_results,

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    "confusion_matrix":
        cm.tolist(),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp)
}


# ============================================================
# 15. WRITE JSON FILE
# ============================================================

RESULTS_PATH = (
    BACKEND_DIR /
    "InternTrust_eval_results.json"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n============================================================")
print("RESULTS SAVED")
print("============================================================")

print(RESULTS_PATH)

print("\n============================================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("============================================================")