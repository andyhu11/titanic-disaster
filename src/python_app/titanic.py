"""
Titanic survival prediction with logistic regression (Python).

Reads train.csv / test.csv from DATA_DIR (default /app/data, mounted by Docker),
prints every data change, trains a logistic regression model, reports accuracy,
and writes predictions for test.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
RANDOM_STATE = 42
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "Title", "FamilySize", "IsAlone", "HasCabin"]
CATEGORICAL = ["Sex", "Embarked", "Title"]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {path} not found.\n"
                 "Download train.csv and test.csv from "
                 "https://www.kaggle.com/competitions/titanic/data into src/data/ "
                 "and mount that folder with -v (see README).")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(names):
    title = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title.where(title.isin(common), "Rare")


def fit_imputation_values(train):
    """Learn every fill value from the TRAINING data only."""
    titles = extract_title(train["Name"])
    return {
        "age_by_title": train.groupby(titles)["Age"].median().to_dict(),
        "age_overall": train["Age"].median(),
        "fare_by_class": train.groupby("Pclass")["Fare"].median().to_dict(),
        "embarked_mode": train["Embarked"].mode()[0],
    }


def preprocess(df, stats, label):
    """Apply the same cleaning steps to any dataset, printing each change."""
    df = df.copy()
    print(f"\n--- Cleaning {label} ---")

    # 1. Title from Name
    df["Title"] = extract_title(df["Name"])
    print(f"[ADD] Title extracted from Name -> counts: "
          f"{df['Title'].value_counts().to_dict()}")

    # 2. Age: fill by median age of the passenger's title
    n_missing = df["Age"].isna().sum()
    fill = df["Title"].map(stats["age_by_title"]).fillna(stats["age_overall"])
    df["Age"] = df["Age"].fillna(fill)
    print(f"[FILL] Age: {n_missing} missing values filled with the median age per Title "
          f"{ {k: round(v, 1) for k, v in stats['age_by_title'].items()} }")

    # 3. Fare: fill by median fare of the passenger class
    n_missing = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(df["Pclass"].map(stats["fare_by_class"]))
    print(f"[FILL] Fare: {n_missing} missing values filled with the median Fare per Pclass")

    # 4. Embarked: fill with the most common port
    n_missing = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])
    print(f"[FILL] Embarked: {n_missing} missing values filled with mode "
          f"'{stats['embarked_mode']}'")

    # 5. Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[ADD] FamilySize = SibSp + Parch + 1 (mean {df['FamilySize'].mean():.2f}); "
          f"IsAlone = 1 if FamilySize == 1 ({df['IsAlone'].sum()} passengers alone)")

    # 6. Cabin -> HasCabin (Cabin itself is mostly missing)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"[ADD] HasCabin = 1 if a cabin is recorded ({df['HasCabin'].sum()} passengers)")

    # 7. Drop columns not used by the model
    dropped = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
    print(f"[DROP] Columns not used as features: {dropped}")

    print(f"[CHECK] Missing values left in features: "
          f"{int(df[FEATURES].isna().sum().sum())}")
    return df


def encode(df, columns=None):
    """One-hot encode categoricals; align to the training columns if given."""
    X = pd.get_dummies(df[FEATURES], columns=CATEGORICAL, drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def main():
    section("STEP 0 - Environment check")
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
    print(f"Reading data from: {DATA_DIR}")

    # ------------------------------------------------------------------ #
    section("STEP 1 - Load train.csv")
    train = load_csv("train.csv")
    print("\nFirst 5 rows:")
    print(train.head().to_string())
    print("\nMissing values per column:")
    print(train.isna().sum()[lambda s: s > 0].to_string())

    # ------------------------------------------------------------------ #
    section("STEP 2 - Explore: a few interesting notes")
    print(f"Overall survival rate: {train['Survived'].mean():.1%}")
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().map("{:.1%}".format).to_string())
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().map("{:.1%}".format).to_string())
    kids = train["Age"] < 16
    print(f"\nSurvival rate for children (<16): {train.loc[kids, 'Survived'].mean():.1%} "
          f"vs adults: {train.loc[train['Age'] >= 16, 'Survived'].mean():.1%}")

    # ------------------------------------------------------------------ #
    section("STEP 3 - Adjust the data (training set)")
    stats = fit_imputation_values(train)
    train_clean = preprocess(train, stats, "train.csv")
    X = encode(train_clean)
    y = train_clean["Survived"]
    print(f"\n[ENCODE] One-hot encoded {CATEGORICAL} -> {X.shape[1]} model features:")
    print(f"         {list(X.columns)}")

    # ------------------------------------------------------------------ #
    section("STEP 4 - Logistic regression on the training set")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Split train.csv 80/20 (stratified): {len(X_tr)} training rows, "
          f"{len(X_val)} validation rows")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    print("Model: StandardScaler -> LogisticRegression (max_iter=1000)")

    coefs = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
    print("\nCoefficients (standardized; + raises, - lowers survival odds):")
    print(coefs.round(3).to_string())

    # ------------------------------------------------------------------ #
    section("STEP 5 - Accuracy on the training set")
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Training accuracy   (80% of train.csv): {train_acc:.4f}")
    print(f"Validation accuracy (20% held out):     {val_acc:.4f}")

    model.fit(X, y)
    full_acc = accuracy_score(y, model.predict(X))
    print(f"Refit on 100% of train.csv -> training accuracy: {full_acc:.4f}")

    # ------------------------------------------------------------------ #
    section("STEP 6 - Load test.csv and predict")
    test = load_csv("test.csv")
    test_clean = preprocess(test, stats, "test.csv (using values learned from train.csv)")
    X_test = encode(test_clean, columns=X.columns)
    test_pred = model.predict(X_test)
    print(f"\nPredicted survivors in test.csv: {test_pred.sum()} of {len(test_pred)} "
          f"({test_pred.mean():.1%})")

    out_path = os.path.join(DATA_DIR, "predictions_python.csv")
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}) \
        .to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path} (Kaggle submission format)")

    # ------------------------------------------------------------------ #
    section("STEP 7 - Accuracy on the test set")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], test_pred):.4f}")
    else:
        print("test.csv has NO 'Survived' column - Kaggle keeps the true labels hidden,")
        print("so test accuracy cannot be computed locally.")
        print(f"Best local estimate (held-out validation accuracy): {val_acc:.4f}")
        print("To get the real test score, submit predictions_python.csv on Kaggle.")

    section("DONE")


if __name__ == "__main__":
    main()
