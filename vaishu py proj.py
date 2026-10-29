# ============================================================
# PROJECT: GODAVARI RIVER WATER QUALITY ANALYSIS
# AND CLASSIFICATION
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. PROJECT INTRODUCTION
# ============================================================

print("=" * 65)
print("GODAVARI RIVER WATER QUALITY ANALYSIS AND CLASSIFICATION")
print("=" * 65)

print("""
PROJECT OBJECTIVES
1. Load and clean the river water quality dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Analyze temperature, D.O., pH, B.O.D., and other measurements.
4. Train and compare six classification models.
5. Evaluate models using classification metrics.
6. Apply K-Means clustering.
7. Apply PCA for dimensionality reduction.
8. Save charts, predictions, and reports.
""")


# ============================================================
# 2. LOAD THE ACTUAL DATASET
# ============================================================

file_path = r"C:\Users\adija\Downloads\water_quality_of_river_godavari-2014.csv"

if not file_path:
    file_path = "water_quality_of_river_godavari-2014(2).csv"

if not Path(file_path).exists():
    raise FileNotFoundError(
        f"\nDataset not found: {file_path}\n"
        "Enter the correct CSV path or place the dataset "
        "in the same folder as this Python file."
    )

df = pd.read_csv(file_path, encoding="cp1252")

print("\nOriginal dataset shape:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nFirst five records:")
print(df.head())


# ============================================================
# 3. CLEAN COLUMN NAMES AND DATA
# ============================================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace("\n", " ", regex=False)
)

before = len(df)

df = df.drop_duplicates().copy()

print("\nDuplicate records removed:", before - len(df))

# Replace common missing-value indicators.
df = df.replace(
    ["-", "NA", "NAN", "N/A", "nil", "Nil", "NIL", ""],
    np.nan
)

# Keep location and state as categorical metadata.
metadata_columns = [
    col for col in ["STATION CODE", "LOCATIONS", "STATE"]
    if col in df.columns
]

# Convert measurement columns into numeric values.
for col in df.columns:
    if col not in ["LOCATIONS", "STATE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nDataset shape after initial cleaning:", df.shape)

print("\nMissing values in each column:")
print(df.isnull().sum())


# ============================================================
# 4. IDENTIFY THE ACTUAL TARGET COLUMNS
# ============================================================

do_column = next(
    (
        col for col in df.columns
        if "D.O." in col.upper() and "MEAN" in col.upper()
    ),
    None
)

bod_column = next(
    (
        col for col in df.columns
        if "B.O.D." in col.upper() and "MEAN" in col.upper()
    ),
    None
)

ph_column = next(
    (
        col for col in df.columns
        if "PH" in col.upper() and "MEAN" in col.upper()
    ),
    None
)

temperature_column = next(
    (
        col for col in df.columns
        if "TEMPERATURE" in col.upper() and "MEAN" in col.upper()
    ),
    None
)

if do_column is None or bod_column is None:
    raise ValueError(
        "The mean D.O. or mean B.O.D. column was not found. "
        "Check the column names printed above."
    )

print("\nSelected columns:")
print("Dissolved oxygen:", do_column)
print("Biological oxygen demand:", bod_column)
print("pH:", ph_column)
print("Temperature:", temperature_column)


# ============================================================
# 5. CREATE OUTPUT FOLDER
# ============================================================

output_dir = Path("Godavari_Water_Quality_Results")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. CREATE WATER QUALITY CLASS
# ============================================================

# Educational project rule:
# Class 1: Mean D.O. >= 5.0 AND mean B.O.D. <= 3.0
# Class 0: Does not satisfy both conditions.
#
# This is a project-defined label, not an official assessment
# of whether water is safe to drink.

df = df.dropna(
    subset=[do_column, bod_column]
).copy()

df["Water_Quality_Class"] = (
    (df[do_column] >= 5.0) &
    (df[bod_column] <= 3.0)
).astype(int)

df["Water_Quality_Label"] = df[
    "Water_Quality_Class"
].map({
    0: "Does not meet project rule",
    1: "Meets project rule"
})

print("\nWater quality class distribution:")
print(df["Water_Quality_Label"].value_counts())

print("\nClass 0: Does not meet the project-defined rule")
print("Class 1: Meets the project-defined rule")

df.to_csv(
    output_dir / "cleaned_water_quality_data.csv",
    index=False
)


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

print("\nDESCRIPTIVE STATISTICS")
print(numeric_df.describe().round(3))

numeric_df.describe().to_csv(
    output_dir / "descriptive_statistics.csv"
)


# ============================================================
# 8. EDA: WATER QUALITY CLASS DISTRIBUTION
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Water_Quality_Label",
    order=[
        "Does not meet project rule",
        "Meets project rule"
    ]
)

plt.title("Godavari River Water Quality Class Distribution")
plt.xlabel("Water Quality Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    output_dir / "01_class_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 9. EDA: TEMPERATURE DISTRIBUTION
# ============================================================

if temperature_column is not None:
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=temperature_column,
        kde=True
    )

    plt.title("Distribution of Mean Water Temperature")
    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        output_dir / "02_temperature_distribution.png",
        dpi=160
    )

    plt.close()


# ============================================================
# 10. EDA: DISSOLVED OXYGEN DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x=do_column,
    kde=True
)

plt.title("Distribution of Mean Dissolved Oxygen")
plt.xlabel("Mean D.O. (mg/l)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    output_dir / "03_do_distribution.png",
    dpi=160
)

plt.close()


# ============================================================
# 11. EDA: B.O.D. BY WATER QUALITY CLASS
# ============================================================

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Water_Quality_Label",
    y=bod_column
)

plt.title("Mean B.O.D. by Water Quality Class")
plt.xlabel("Water Quality Class")
plt.ylabel("Mean B.O.D. (mg/l)")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    output_dir / "04_bod_boxplot.png",
    dpi=160
)

plt.close()


# ============================================================
# 12. EDA: PH DISTRIBUTION
# ============================================================

if ph_column is not None:
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=ph_column,
        kde=True
    )

    plt.title("Distribution of Mean pH")
    plt.xlabel("Mean pH")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        output_dir / "05_ph_distribution.png",
        dpi=160
    )

    plt.close()


# ============================================================
# 13. EDA: WATER QUALITY BY LOCATION
# ============================================================

if "LOCATIONS" in df.columns:
    location_data = df[
        ["LOCATIONS", do_column]
    ].dropna().sort_values(
        by=do_column,
        ascending=True
    )

    if not location_data.empty:
        plt.figure(
            figsize=(12, max(6, len(location_data) * 0.35))
        )

        sns.barplot(
            data=location_data,
            x=do_column,
            y="LOCATIONS"
        )

        plt.title("Mean Dissolved Oxygen by River Location")
        plt.xlabel("Mean D.O. (mg/l)")
        plt.ylabel("Location")
        plt.tight_layout()

        plt.savefig(
            output_dir / "06_do_by_location.png",
            dpi=160
        )

        plt.close()


# ============================================================
# 14. EDA: CORRELATION HEATMAP
# ============================================================

# Exclude the station code because it is an identifier,
# not a meaningful water-quality measurement.

measurement_columns = [
    col for col in numeric_df.columns
    if col != "STATION CODE"
]

# Use measurement columns only, excluding generated target columns.
measurement_columns = [
    col for col in measurement_columns
    if col not in [
        "Water_Quality_Class"
    ]
]

if len(measurement_columns) >= 2:
    correlation_matrix = df[
        measurement_columns
    ].corr()

    plt.figure(figsize=(18, 12))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title("Water Quality Measurement Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        output_dir / "07_correlation_heatmap.png",
        dpi=160
    )

    plt.close()

    correlation_matrix.to_csv(
        output_dir / "correlation_matrix.csv"
    )


# ============================================================
# 15. SELECT PREDICTOR FEATURES
# ============================================================

# Exclude:
# - D.O. columns
# - B.O.D. columns
# - station code
# - generated target columns
#
# D.O. and B.O.D. define the target, so including them as
# predictors would cause target leakage.

excluded_columns = []

for col in df.columns:
    upper_col = col.upper()

    if (
        "D.O." in upper_col
        or "B.O.D." in upper_col
        or col in [
            "STATION CODE",
            "Water_Quality_Class",
            "Water_Quality_Label",
            "LOCATIONS",
            "STATE"
        ]
    ):
        excluded_columns.append(col)

feature_columns = [
    col for col in df.select_dtypes(include=np.number).columns
    if col not in excluded_columns
]

X = df[feature_columns].copy()
y = df["Water_Quality_Class"].copy()

# Remove columns that have no usable values.
X = X.dropna(axis=1, how="all")

# Remove constant columns.
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] == 0:
    raise ValueError(
        "No suitable numeric predictors were found."
    )

print("\nPredictor columns:")
print(X.columns.tolist())

print("\nNumber of predictor features:", X.shape[1])


# ============================================================
# 16. CHECK CLASS COUNTS
# ============================================================

class_counts = y.value_counts()

print("\nTarget class counts:")
print(class_counts)

if len(class_counts) != 2 or class_counts.min() < 2:
    raise ValueError(
        "The target must contain two classes with at least "
        "two records in each class."
    )


# ============================================================
# 17. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 18. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=min(3, len(X_train))
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42
    ),

    "Support Vector Machine": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 19. TRAIN AND EVALUATE ALL MODELS
# ============================================================

trained_models = {}
predictions = {}
results = []

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training model:", model_name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(
            y_test,
            y_probability
        )
    else:
        auc = np.nan

    loss = log_loss(
        y_test,
        y_probability,
        labels=[0, 1]
    )

    trained_models[model_name] = pipeline
    predictions[model_name] = y_pred

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "Log Loss": loss
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print(
        "ROC-AUC  :",
        round(auc, 4) if not np.isnan(auc) else "N/A"
    )
    print("Log Loss :", round(loss, 4))


# ============================================================
# 20. MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

comparison_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(comparison_df.round(4).to_string(index=False))

best_model_name = comparison_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print("\nBest model by test F1-score:", best_model_name)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=[
        "Does not meet project rule",
        "Meets project rule"
    ],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

(output_dir / "classification_report.txt").write_text(
    report,
    encoding="utf-8"
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    {
        "Actual_Class": y_test.values,
        "Predicted_Class": best_predictions
    },
    index=y_test.index
)

prediction_df.index.name = "Original_Row_Index"

prediction_df["Actual_Label"] = prediction_df[
    "Actual_Class"
].map({
    0: "Does not meet project rule",
    1: "Meets project rule"
})

prediction_df["Predicted_Label"] = prediction_df[
    "Predicted_Class"
].map({
    0: "Does not meet project rule",
    1: "Meets project rule"
})

if "LOCATIONS" in df.columns:
    prediction_df["Location"] = df.loc[
        prediction_df.index,
        "LOCATIONS"
    ]

prediction_df.to_csv(
    output_dir / "test_set_predictions.csv"
)


# ============================================================
# 23. MODEL COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

comparison_plot = comparison_df.set_index("Model")[metrics]

comparison_plot.plot(
    kind="bar",
    figsize=(13, 6),
    ylim=(0, 1.05)
)

plt.title("Classification Model Performance Comparison")
plt.xlabel("Classification Model")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    output_dir / "08_model_comparison.png",
    dpi=160
)

plt.close()


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Does not meet rule",
        "Meets rule"
    ],
    yticklabels=[
        "Does not meet rule",
        "Meets rule"
    ]
)

plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    output_dir / "09_confusion_matrix.png",
    dpi=160
)

plt.close()


# ============================================================
# 25. K-MEANS CLUSTERING
# ============================================================

print("\nPerforming K-Means clustering...")

cluster_imputer = SimpleImputer(
    strategy="median"
)

cluster_data = cluster_imputer.fit_transform(X)

cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)

number_of_clusters = min(3, len(df))

kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    cluster_scaled
)

clustered_df = df.copy()
clustered_df["KMeans_Cluster"] = cluster_labels

clustered_df.to_csv(
    output_dir / "kmeans_clustered_water_quality.csv",
    index=False
)

cluster_summary = clustered_df.groupby(
    "KMeans_Cluster"
)[X.columns].mean(numeric_only=True)

cluster_summary.to_csv(
    output_dir / "kmeans_cluster_summary.csv"
)

print("\nK-Means cluster summary:")
print(cluster_summary.round(3))


# Cluster size chart.
plt.figure(figsize=(8, 5))

sns.countplot(
    x=cluster_labels
)

plt.title("K-Means Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    output_dir / "10_kmeans_cluster_sizes.png",
    dpi=160
)

plt.close()


# ============================================================
# 26. PCA DIMENSIONALITY REDUCTION
# ============================================================

if cluster_scaled.shape[1] >= 2:

    pca = PCA(n_components=2)

    pca_coordinates = pca.fit_transform(
        cluster_scaled
    )

    pca_df = pd.DataFrame({
        "Principal Component 1": pca_coordinates[:, 0],
        "Principal Component 2": pca_coordinates[:, 1],
        "Cluster": cluster_labels
    })

    pca_df.to_csv(
        output_dir / "pca_coordinates.csv",
        index=False
    )

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=pca_df,
        x="Principal Component 1",
        y="Principal Component 2",
        hue="Cluster",
        palette="viridis",
        s=100
    )

    variance = pca.explained_variance_ratio_.sum()

    plt.title(
        f"PCA Visualization of K-Means Clusters "
        f"({variance:.1%} variance explained)"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "11_pca_visualization.png",
        dpi=160
    )

    plt.close()

    print("\nPCA explained variance:")
    print(pca.explained_variance_ratio_)

else:
    print(
        "\nPCA skipped because at least two predictor "
        "features are required."
    )


# ============================================================
# 27. SAVE PROJECT SUMMARY
# ============================================================

best_row = comparison_df.iloc[0]

summary = f"""
GODAVARI RIVER WATER QUALITY ANALYSIS AND CLASSIFICATION

Dataset records after cleaning: {len(df)}
Predictor features: {X.shape[1]}
Training records: {len(X_train)}
Testing records: {len(X_test)}

Target rule:
Mean D.O. >= 5.0 and mean B.O.D. <= 3.0.

Class 1 meets this educational project rule.
Class 0 does not meet both conditions.
This rule does not certify drinking-water safety.

Best model by test F1-score: {best_model_name}

Accuracy: {best_row['Accuracy']:.4f}
Precision: {best_row['Precision']:.4f}
Recall: {best_row['Recall']:.4f}
F1 Score: {best_row['F1 Score']:.4f}
ROC-AUC: {best_row['ROC-AUC']:.4f}
Log Loss: {best_row['Log Loss']:.4f}

Limitations:
The dataset contains only 35 original records, so performance
metrics can be unstable and sensitive to the train/test split.
The target is derived from D.O. and B.O.D. thresholds.
Those columns are excluded from the classifier's predictors.
K-Means and PCA are exploratory analyses.
"""

(output_dir / "project_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# ============================================================
# 28. DISPLAY ALL OUTPUT FILES
# ============================================================

print("\n" + "=" * 65)
print("PROJECT COMPLETED")
print("=" * 65)

print("\nResults folder:")
print(output_dir.resolve())

print("\nGenerated files:")

for output_file in sorted(output_dir.iterdir()):
    print("-", output_file.name)

print("\nOpen the Godavari_Water_Quality_Results folder")
print("to view your charts, reports, and CSV results.")

print("\nImportant:")
print("Use the actual model_comparison.csv values in your report.")
print("Do not treat this small dataset's scores as definitive.")
